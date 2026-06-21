/*
 * FALL - SSD Parallel Scan Kernel (Mamba‑2)
 * Implements the associative scan for state space duality.
 * Uses Blelloch scan algorithm for O(log N) parallel time.
 */
#include <cuda_fp16.h>

__global__ void ssd_scan_fwd(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ Y,
    const int batch_size,
    const int seq_len,
    const int d_state,
    const int d_model
) {
    extern __shared__ float shared[];
    
    int bid = blockIdx.x;
    int dim_idx = blockIdx.y;
    
    // Load A, B, C into shared memory
    float* A_shared = shared;
    float* B_shared = shared + seq_len * d_state;
    
    // Up‑sweep (reduce)
    for (int stride = 1; stride < seq_len; stride *= 2) {
        if (threadIdx.x % (2 * stride) == 0) {
            int idx = threadIdx.x;
            // Combine adjacent states: h_new = A[idx+stride] * A[idx] * h
            for (int s = 0; s < d_state; s++) {
                A_shared[(idx + 2*stride - 1) * d_state + s] *= 
                    A_shared[(idx + stride - 1) * d_state + s];
            }
        }
        __syncthreads();
    }
    
    // Down‑sweep (scan)
    // Clear last element
    if (threadIdx.x == seq_len - 1) {
        for (int s = 0; s < d_state; s++) {
            A_shared[threadIdx.x * d_state + s] = 0.0f;
        }
    }
    __syncthreads();
    
    for (int stride = seq_len / 2; stride > 0; stride /= 2) {
        if (threadIdx.x % (2 * stride) == 0) {
            int idx = threadIdx.x;
            float temp[STATE_DIM];
            for (int s = 0; s < d_state; s++) {
                temp[s] = A_shared[(idx + stride - 1) * d_state + s];
                A_shared[(idx + stride - 1) * d_state + s] = 
                    A_shared[(idx + 2*stride - 1) * d_state + s];
                A_shared[(idx + 2*stride - 1) * d_state + s] *= temp[s];
            }
        }
        __syncthreads();
    }
    
    // Compute Y = C @ h for each position
    for (int t = 0; t < seq_len; t++) {
        float acc = 0.0f;
        for (int s = 0; s < d_state; s++) {
            acc += C[t * d_state + s] * A_shared[t * d_state + s];
        }
        Y[t] = acc;
    }
}