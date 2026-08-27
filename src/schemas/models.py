from pydantic import BaseModel

class DatasetEntry(BaseModel):
    query: str
    expected_context: str
    expected_answer: str

class EvaluationResult(BaseModel):
    passed: bool
    latency: float
    tokens: int
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    correctness: float = 0.0
