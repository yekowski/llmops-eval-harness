from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class DatasetEntry(BaseModel):
    query: str
    expected_context: str = ""
    expected_answer: str = ""
    retrieved_contexts: Optional[List[str]] = None
    ground_truth: Optional[str] = None

class SUTExecutionResult(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

class JudgeGenerationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness: float = Field(..., ge=0.0, le=1.0)
    faithfulness_reasoning: str = ""
    answer_relevance: float = Field(..., ge=0.0, le=1.0)
    answer_relevance_reasoning: str = ""
    correctness: float = Field(..., ge=0.0, le=1.0)
    correctness_reasoning: str = ""

class JudgeRetrievalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_precision: float = Field(..., ge=0.0, le=1.0)
    context_precision_reasoning: str = ""
    context_recall: float = Field(..., ge=0.0, le=1.0)
    context_recall_reasoning: str = ""

class EvaluationResult(BaseModel):
    passed: bool
    latency: float = Field(..., description="System Under Test response latency in seconds")
    tokens: int = Field(default=0, description="Approximate or actual SUT token count")
    sut_prompt_tokens: int = 0
    sut_completion_tokens: int = 0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    sut_cost: float = 0.0
    judge_cost: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    correctness: float = 0.0
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    judge_latency: float = Field(default=0.0, description="Judge evaluation execution latency in seconds")
    judge_mode: Literal["llm", "fallback", "cache"] = "llm"
    retrieval_judge_mode: Optional[Literal["llm", "fallback", "cache"]] = None

class ConcurrencyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_workers: int = Field(default=5, ge=1)
    requests_per_second: float = Field(default=1.0, gt=0.0)

class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: Optional[str] = None
    judge_failure_policy: Literal["fail", "warn", "allow"] = "fail"
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    sla_thresholds: Dict[str, Any] = Field(default_factory=dict)
    sut: Dict[str, Any] = Field(default_factory=dict)
    judge: Dict[str, Any] = Field(default_factory=dict)
    # Optional backward-compatible keys
    slas: Optional[Dict[str, Any]] = None
    fallback_chain: Optional[List[str]] = None
    providers: Optional[Dict[str, Any]] = None
