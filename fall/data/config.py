from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataPipelineConfig:
    # Sources
    github_repos_path: str = "/data/github"
    common_crawl_path: str = "/data/commoncrawl"
    security_corpus_path: str = "/data/security"
    papers_path: str = "/data/papers"
    
    # Processing
    min_doc_length: int = 100
    max_doc_length: int = 1_000_000
    dedup_threshold: float = 0.8
    minhash_permutations: int = 256
    
    # Tokenizer
    vocab_size: int = 256_000
    tokenizer_path: str = "/data/tokenizer"
    
    # Streaming
    kafka_brokers: List[str] = None
    batch_size: int = 64
    seq_len: int = 524_288
    
    # Quality
    min_english_score: float = 0.7
    min_code_score: float = 0.6
    pii_redaction: bool = True