from typing import List, Optional
from pydantic import BaseModel

class DatasetEntry(BaseModel):
    query: str
    expected_context: str = ""
    expected_answer: str = ""
    retrieved_contexts: Optional[List[str]] = None
    ground_truth: Optional[str] = None

class EvaluationResult(BaseModel):
    passed: bool
    latency: float
    tokens: int
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    correctness: float = 0.0
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    judge_latency: float = 0.0
