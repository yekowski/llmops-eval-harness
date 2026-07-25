from pydantic import BaseModel

class DatasetEntry(BaseModel):
    query: str
    expected_context: str
    expected_answer: str

class EvaluationResult(BaseModel):
    passed: bool
    latency: float
    tokens: int
