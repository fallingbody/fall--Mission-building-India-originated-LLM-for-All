/*
 * FALL - Fused Rotary Position Embedding Kernel with YaRN Scaling
 * Applies RoPE in-place with support for 1M+ token sequences.
 * 
 * Compilation:
 * nvcc -O3 --use_fast_math -gencode arch=compute_90,code=sm_90 -c rope_fused.cu
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>

#define WARP_SIZE 32

/*
 * Apply YaRN RoPE to half-precision tensor.
 * 
 * Grid: (batch * n_heads * ceil(seq_len/256), 1, 1)
 * Block: (256, 1, 1)
 */
extern "C" __global__ void rope_fwd_fp16(
    half* __restrict__ x,                    // [B, H, L, D] - modified in-place
    const float* __restrict__ cos_freqs,      // [L, D/2] precomputed cos
    const float* __restrict__ sin_freqs,      // [L, D/2] precomputed sin
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int d_head,
    const int offset                          // position offset for KV cache
) {
    // Each thread processes one (d_head/2) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_pairs = batch_size * num_heads * seq_len * (d_head / 2);
    if (idx >= total_pairs) return;
    
    // Decode indices
    int pair_idx = idx;
    int l = pair_idx / (num_heads * (d_head / 2));           // sequence position
    int remaining = pair_idx % (num_heads * (d_head / 2));
    int h = remaining / (d_head / 2);                         // head index
    int d = remaining % (d_head / 2);                         // dimension pair
    
    // Position in sequence (with offset for KV cache)
    int pos = offset + l;
    
    // Base index into tensor
    int base = l * num_heads * d_head + h * d_head;
    
    // Load cosine and sine values for this position and frequency
    float cos_val = cos_freqs[pos * (d_head / 2) + d];
    float sin_val = sin_freqs[pos * (d_head / 2) + d];
    
    // Load the pair of values
    half x0 = x[base + 2 * d];
    half x1 = x[base + 2 * d + 1];
    
    // Convert to float for computation
    float f0 = __half2float(x0);
    float f1 = __half2float(x1);
    
    // Apply rotation: 
    // x0' = x0 * cos - x1 * sin
    // x1' = x0 * sin + x1 * cos
    float new_x0 = f0 * cos_val - f1 * sin_val;
    float new_x1 = f0 * sin_val + f1 * cos_val;
    
    // Store back
    x[base + 2 * d]     = __float2half(new_x0);
    x[base + 2 * d + 1] = __float2half(new_x1);
}


/*
 * BF16 version of fused RoPE.
 */
extern "C" __global__ void rope_fwd_bf16(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ cos_freqs,
    const float* __restrict__ sin_freqs,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int d_head,
    const int offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = batch_size * num_heads * seq_len * (d_head / 2);
    if (idx >= total_pairs) return;
    
    int l = idx / (num_heads * (d_head / 2));
    int remaining = idx % (num_heads * (d_head / 2));
    int h = remaining / (d_head / 2);
    int d = remaining % (d_head / 2);
    
    int pos = offset + l;
    int base = l * num_heads * d_head + h * d_head;
    
    float cos_val = cos_freqs[pos * (d_head / 2) + d];
    float sin_val = sin_freqs[pos * (d_head / 2) + d];
    
    float f0 = __bfloat162float(x[base + 2 * d]);
    float f1 = __bfloat162float(x[base + 2 * d + 1]);
    
    float new_x0 = f0 * cos_val - f1 * sin_val;
    float new_x1 = f0 * sin_val + f1 * cos_val;
    
    x[base + 2 * d]     = __float2bfloat16(new_x0);
    x[base + 2 * d + 1] = __float2bfloat16(new_x1);
}


/*
 * Vectorized version for better memory coalescing.
 * Each warp processes 32 consecutive pairs.
 */
extern "C" __global__ void rope_fwd_fp16_vectorized(
    half* __restrict__ x,
    const float* __restrict__ cos_freqs,
    const float* __restrict__ sin_freqs,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int d_head,
    const int offset
) {
    // Each block processes one (batch, head) pair
    int b = blockIdx.x / num_heads;
    int h = blockIdx.x % num_heads;
    
    // Each thread processes one position
    int l = blockIdx.y * blockDim.x + threadIdx.x;
    if (l >= seq_len) return;
    
    int pos = offset + l;
    int base = b * num_heads * seq_len * d_head + h * seq_len * d_head + l * d_head;
    
    // Load cos/sin for this position (all frequencies)
    const float* cos_row = cos_freqs + pos * (d_head / 2);
    const float* sin_row = sin_freqs + pos * (d_head / 2);
    
    // Process pairs in a loop
    for (int d = threadIdx.y; d < d_head / 2; d += blockDim.y) {
        float cos_val = cos_row[d];
        float sin_val = sin_row[d];
        
        half x0 = x[base + 2 * d];
        half x1 = x[base + 2 * d + 1];
        
        float f0 = __half2float(x0);
        float f1 = __half2float(x1);
        
        float new_x0 = f0 * cos_val - f1 * sin_val;
        float new_x1 = f0 * sin_val + f1 * cos_val;
        
        x[base + 2 * d]     = __float2half(new_x0);
        x[base + 2 * d + 1] = __float2half(new_x1);
    }
}


/*
 * FP8 version - loads FP8, computes in FP32, stores FP8.
 */
extern "C" __global__ void rope_fwd_fp8(
    __nv_fp8_e4m3* __restrict__ x,
    const float* __restrict__ cos_freqs,
    const float* __restrict__ sin_freqs,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int d_head,
    const int offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = batch_size * num_heads * seq_len * (d_head / 2);
    if (idx >= total_pairs) return;
    
    int l = idx / (num_heads * (d_head / 2));
    int remaining = idx % (num_heads * (d_head / 2));
    int h = remaining / (d_head / 2);
    int d = remaining % (d_head / 2);
    
    int pos = offset + l;
    int base = l * num_heads * d_head + h * d_head;
    
    float cos_val = cos_freqs[pos * (d_head / 2) + d];
    float sin_val = sin_freqs[pos * (d_head / 2) + d];
    
    // FP8 to float
    float f0 = __nv_fp8_e4m3_to_float(x[base + 2 * d]);
    float f1 = __nv_fp8_e4m3_to_float(x[base + 2 * d + 1]);
    
    float new_x0 = f0 * cos_val - f1 * sin_val;
    float new_x1 = f0 * sin_val + f1 * cos_val;
    
    // Float to FP8
    x[base + 2 * d]     = __nv_fp8_e4m3_from_float(new_x0);
    x[base + 2 * d + 1] = __nv_fp8_e4m3_from_float(new_x1);
}


/*
 * Host-side launcher for convenience.
 */
extern "C" void launch_rope_fwd(
    void* x,
    const float* cos_freqs,
    const float* sin_freqs,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_head,
    int offset,
    int dtype  // 0=fp16, 1=bf16, 2=fp8
) {
    int total_pairs = batch_size * num_heads * seq_len * (d_head / 2);
    int block_size = 256;
    int grid_size = (total_pairs + block_size - 1) / block_size;
    
    switch (dtype) {
        case 0:
            rope_fwd_fp16<<<grid_size, block_size>>>(
                (half*)x, cos_freqs, sin_freqs,
                batch_size, num_heads, seq_len, d_head, offset
            );
            break;
        case 1:
            rope_fwd_bf16<<<grid_size, block_size>>>(
                (__nv_bfloat16*)x, cos_freqs, sin_freqs,
                batch_size, num_heads, seq_len, d_head, offset
            );
            break;
        case 2:
            rope_fwd_fp8<<<grid_size, block_size>>>(
                (__nv_fp8_e4m3*)x, cos_freqs, sin_freqs,
                batch_size, num_heads, seq_len, d_head, offset
            );
            break;
    }
}