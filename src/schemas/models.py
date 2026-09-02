from typing import List, Optional
from pydantic import BaseModel, Field

class DatasetEntry(BaseModel):
    query: str
    expected_context: str = ""
    expected_answer: str = ""
    retrieved_contexts: Optional[List[str]] = None
    ground_truth: Optional[str] = None

class EvaluationResult(BaseModel):
    passed: bool
    latency: float = Field(..., description="System Under Test response latency in seconds")
    tokens: int = Field(default=0, description="Approximate response output word/token count")
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    correctness: float = 0.0
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    judge_latency: float = Field(default=0.0, description="Judge evaluation execution latency in seconds")
