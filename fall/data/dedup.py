"""
Deduplication using MinHash and SimHash.
"""
import hashlib
import struct
import numpy as np
from typing import List, Set, Tuple
from collections import defaultdict

class MinHashDeduplicator:
    def __init__(self, num_perm=256, threshold=0.8):
        self.num_perm = num_perm
        self.threshold = threshold
        self.seeds = [hashlib.sha256(str(i).encode()).digest()[:4] for i in range(num_perm)]
        self.seeds = [struct.unpack('<I', s)[0] for s in self.seeds]

    def compute_signature(self, text: str) -> np.ndarray:
        """Compute MinHash signature for a document."""
        shingles = self._get_shingles(text)
        if not shingles:
            return np.full(self.num_perm, np.inf)

        signature = np.full(self.num_perm, np.inf)
        for shingle in shingles:
            shingle_hash = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
            for i, seed in enumerate(self.seeds):
                h = (shingle_hash ^ seed) % (2**32)
                if h < signature[i]:
                    signature[i] = h
        return signature

    def estimate_jaccard(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        return np.mean(sig1 == sig2)

    def _get_shingles(self, text: str, k=3) -> Set[str]:
        """Extract character n-grams from text."""
        text = text.lower()
        return {text[i:i+k] for i in range(len(text) - k + 1)}


class LSHIndex:
    """Locality-Sensitive Hashing for approximate nearest neighbor search."""
    def __init__(self, num_bands=32, rows_per_band=8):
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.buckets = [defaultdict(list) for _ in range(num_bands)]

    def insert(self, doc_id: str, signature: np.ndarray):
        for band in range(self.num_bands):
            start = band * self.rows_per_band
            end = start + self.rows_per_band
            band_sig = tuple(signature[start:end].astype(int))
            bucket_key = hash(band_sig)
            self.buckets[band][bucket_key].append(doc_id)

    def query_candidates(self, signature: np.ndarray) -> Set[str]:
        candidates = set()
        for band in range(self.num_bands):
            start = band * self.rows_per_band
            end = start + self.rows_per_band
            band_sig = tuple(signature[start:end].astype(int))
            bucket_key = hash(band_sig)
            candidates.update(self.buckets[band].get(bucket_key, []))
        return candidates


class DedupManager:
    def __init__(self, config):
        self.minhash = MinHashDeduplicator(
            num_perm=config.minhash_permutations,
            threshold=config.dedup_threshold,
        )
        self.lsh = LSHIndex()
        self.documents = {}

    def is_duplicate(self, doc_id: str, text: str) -> Tuple[bool, float]:
        """Check if document is a duplicate. Returns (is_dup, max_similarity)."""
        sig = self.minhash.compute_signature(text)
        candidates = self.lsh.query_candidates(sig)
        max_sim = 0.0
        for cand_id in candidates:
            cand_sig = self.documents.get(cand_id)
            if cand_sig is not None:
                sim = self.minhash.estimate_jaccard(sig, cand_sig)
                max_sim = max(max_sim, sim)
        return max_sim > self.minhash.threshold, max_sim

    def add(self, doc_id: str, text: str):
        """Add document to the dedup index."""
        sig = self.minhash.compute_signature(text)
        self.documents[doc_id] = sig
        self.lsh.insert(doc_id, sig)