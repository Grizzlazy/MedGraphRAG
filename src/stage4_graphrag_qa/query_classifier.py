"""
Stage 4 — Query Classifier
Classifies incoming questions as local / global / hybrid / aggregation
using a fine-tuned RoBERTa model.
"""

from pathlib import Path
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from loguru import logger

QUERY_TYPES = ["local", "global", "hybrid", "aggregation"]

# Rule-based keywords for fallback classification
_LOCAL_KEYWORDS = [
    "patient", "bệnh nhân", "was diagnosed", "prescribed",
    "what medication", "what diagnosis",
]
_GLOBAL_KEYWORDS = [
    "how many", "most common", "average", "trend", "overall",
    "compare", "percentage", "statistics", "all patients",
    "tổng", "phần trăm", "xu hướng", "tất cả",
]


class QueryClassifier:
    """Classifies clinical questions to select the appropriate retrieval strategy."""

    def __init__(self, model_path: str | None = None):
        """
        Args:
            model_path: Path to fine-tuned RoBERTa checkpoint.
                        If None, uses rule-based fallback.
        """
        self.model = None
        self.tokenizer = None

        if model_path and Path(model_path).exists():
            try:
                self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
                self.model = RobertaForSequenceClassification.from_pretrained(
                    model_path, num_labels=len(QUERY_TYPES),
                )
                self.model.eval()
                logger.info(f"QueryClassifier loaded from {model_path}")
            except Exception as exc:
                logger.warning(f"Failed to load RoBERTa classifier: {exc}. Using rule-based.")
        else:
            logger.info("QueryClassifier running in rule-based mode (no model checkpoint)")

    def classify(self, question: str) -> str:
        """
        Classify a question into one of: local, global, hybrid, aggregation.

        Uses RoBERTa if available, otherwise falls back to keyword rules.
        """
        if self.model and self.tokenizer:
            return self._classify_model(question)
        return self._classify_rules(question)

    @torch.no_grad()
    def _classify_model(self, question: str) -> str:
        inputs = self.tokenizer(
            question, return_tensors="pt", truncation=True, max_length=128,
        )
        logits = self.model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        return QUERY_TYPES[pred]

    def _classify_rules(self, question: str) -> str:
        q_lower = question.lower()

        is_local = any(kw in q_lower for kw in _LOCAL_KEYWORDS)
        is_global = any(kw in q_lower for kw in _GLOBAL_KEYWORDS)

        if is_local and is_global:
            return "hybrid"
        elif is_global:
            return "global"
        elif is_local:
            return "local"
        else:
            return "local"  # Default
