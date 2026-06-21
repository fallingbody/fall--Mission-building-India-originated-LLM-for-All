"""
Data processing pipeline for FALL.
Cleans, filters, and normalizes raw text from all sources.
"""
import re
import html
import ftfy
import unicodedata
from typing import Tuple, Optional, List
from dataclasses import dataclass
import numpy as np

@dataclass
class ProcessedDocument:
    text: str
    source: str
    language: str
    quality_score: float
    token_count: int
    has_pii: bool
    metadata: dict

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.pii_patterns = self._compile_pii_patterns()

    def _compile_pii_patterns(self):
        return {
            "email": re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
            "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        }

    def process(self, raw_text: str, source: str = "unknown") -> Optional[ProcessedDocument]:
        """Process a raw document through the entire pipeline."""
        # Step 1: Fix encoding
        text = ftfy.fix_text(raw_text)
        text = html.unescape(text)
        text = unicodedata.normalize("NFKC", text)

        # Step 2: Basic cleaning
        text = self._clean_whitespace(text)
        text = self._remove_control_chars(text)

        # Step 3: Language detection
        language = self._detect_language(text)
        if not language:
            return None

        # Step 4: Quality filtering
        quality_score = self._compute_quality(text)
        if quality_score < self.config.min_english_score and language == "en":
            return None

        # Step 5: PII detection
        has_pii = self._detect_pii(text)
        if has_pii and self.config.pii_redaction:
            text = self._redact_pii(text)

        # Step 6: Length check
        token_count = len(text.split())
        if token_count < self.config.min_doc_length:
            return None
        if token_count > self.config.max_doc_length:
            text = self._truncate(text, self.config.max_doc_length)

        return ProcessedDocument(
            text=text,
            source=source,
            language=language,
            quality_score=quality_score,
            token_count=token_count,
            has_pii=has_pii,
            metadata={"processed_at": None}
        )

    def _clean_whitespace(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _remove_control_chars(self, text: str) -> str:
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    def _detect_language(self, text: str) -> Optional[str]:
        # Simplified — use fasttext or lingua in production
        if len(text) < 50:
            return None
        return "en"  # Placeholder

    def _compute_quality(self, text: str) -> float:
        scores = []
        # Average word length
        words = text.split()
        if words:
            avg_len = np.mean([len(w) for w in words[:1000]])
            scores.append(min(avg_len / 6.0, 1.0))
        # Unique word ratio
        if len(words) > 100:
            unique_ratio = len(set(words[:1000])) / min(len(words[:1000]), 1000)
            scores.append(unique_ratio)
        # No excessive repetition
        if len(words) > 50:
            max_repeat = max([words[i:i+5].count(w) for i, w in enumerate(words[:100])] or [1])
            scores.append(1.0 / max(1, max_repeat))
        return np.mean(scores) if scores else 0.0

    def _detect_pii(self, text: str) -> bool:
        for pattern in self.pii_patterns.values():
            if pattern.search(text):
                return True
        return False

    def _redact_pii(self, text: str) -> str:
        for name, pattern in self.pii_patterns.items():
            text = pattern.sub(f"[{name.upper()}]", text)
        return text

    def _truncate(self, text: str, max_tokens: int) -> str:
        words = text.split()
        return ' '.join(words[:max_tokens])