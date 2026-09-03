import json
import asyncio
from typing import Optional, List
from pydantic import ValidationError
from src.evaluation.prompts.judge_templates import JUDGE_PROMPT_TEMPLATE, RETRIEVAL_JUDGE_PROMPT_TEMPLATE
from src.utils.cache import EvalCache
from src.utils.helpers import strip_markdown_json, resolve_error_status_code
from src.providers.base import LLMProvider, ProviderResponse
from src.providers.gemini import GeminiProvider
from src.utils.pricing import calculate_token_cost
from src.schemas.models import JudgeGenerationOutput, JudgeRetrievalOutput

class LLMJudge:
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        api_key: Optional[str] = None,
        model: str = "gemini-3.5-flash",
        cache: Optional[EvalCache] = None,
        sla_thresholds: Optional[dict] = None
    ):
        self.cache = cache
        self.total_cost = 0.0
        self.total_judge_prompt_tokens = 0
        self.total_judge_completion_tokens = 0
        self._cost_lock = asyncio.Lock()
        self.provider = provider or GeminiProvider(api_key=api_key, model=model)
        self.sla_thresholds = sla_thresholds or {}

    async def close(self) -> None:
        """Closes underlying provider connections."""
        if hasattr(self.provider, "close"):
            try:
                await self.provider.close()
            except Exception:
                pass

    def _get_provider_class(self) -> str:
        if hasattr(self.provider, "provider_name") and self.provider.provider_name:
            return self.provider.provider_name
        if hasattr(self.provider, "active_provider"):
            act = self.provider.active_provider
            return getattr(act, "provider_name", act.__class__.__name__)
        return getattr(self.provider, "provider_name", self.provider.__class__.__name__)

    def _compute_passed(self, faithfulness: float, relevance: float, correctness: float) -> bool:
        """Dynamically computes SLA pass status against currently configured thresholds."""
        min_faith = self.sla_thresholds.get("min_faithfulness", self.sla_thresholds.get("faithfulness", 0.8))
        min_rel = self.sla_thresholds.get("min_relevance", self.sla_thresholds.get("relevance", 0.8))
        min_corr = self.sla_thresholds.get("min_correctness", self.sla_thresholds.get("correctness", 0.8))
        return (faithfulness >= min_faith and relevance >= min_rel and correctness >= min_corr)

    def get_cached_evaluation(
        self,
        query: str,
        context: str,
        expected_answer: str,
        generated_answer: str,
        ground_truth: str = ""
    ) -> Optional[dict]:
        """Check cache for evaluation result, dynamically evaluating pass against active SLA thresholds."""
        if not self.cache:
            return None
        q = query if query is not None else ""
        model_name = self.provider.model
        prov_cls = self._get_provider_class()
        cached = self.cache.get(
            generated_answer=generated_answer,
            query=q,
            context=context,
            expected_answer=expected_answer,
            ground_truth=ground_truth,
            provider_class=prov_cls,
            model=model_name,
            prompt_template=JUDGE_PROMPT_TEMPLATE,
            prompt_template_version="v2"
        )
        if cached is None and hasattr(self.provider, "providers"):
            for p in self.provider.providers:
                p_cls = getattr(p, "provider_name", p.__class__.__name__)
                p_model = getattr(p, "model", "")
                c = self.cache.get(
                    generated_answer=generated_answer,
                    query=q,
                    context=context,
                    expected_answer=expected_answer,
                    ground_truth=ground_truth,
                    provider_class=p_cls,
                    model=p_model,
                    prompt_template=JUDGE_PROMPT_TEMPLATE,
                    prompt_template_version="v2"
                )
                if c is not None:
                    cached = c
                    break

        if cached is not None:
            res = dict(cached)
            res["passed"] = self._compute_passed(
                res.get("faithfulness", 0.0),
                res.get("answer_relevance", 0.0),
                res.get("correctness", 0.0)
            )
            res["judge_mode"] = "cache"
            res["judge_provenance"] = cached.get("judge_provenance", "unknown")
            return res
        return None

    def _is_local_fallback_required(self) -> bool:
        """Determines if the judge should use local deterministic grading because no remote API key is configured."""
        if not self.provider:
            return True
        if hasattr(self.provider, "api_key"):
            return not bool(self.provider.api_key)
        if hasattr(self.provider, "providers"):
            has_valid_remote_key = False
            for p in self.provider.providers:
                if p.__class__.__name__ == "MockProvider":
                    continue
                if getattr(p, "execution_mode", "remote") == "local":
                    continue
                if getattr(p, "api_key", None) and getattr(p, "api_key") != "local":
                    has_valid_remote_key = True
                    break
            return not has_valid_remote_key
        return False

    async def evaluate(
        self,
        context: str,
        expected_answer: str,
        generated_answer: str,
        query: Optional[str] = None,
        ground_truth: Optional[str] = None
    ) -> dict:
        """Evaluates a generated response against context and expected answer."""
        q = query if query is not None else expected_answer
        gt = ground_truth or expected_answer
        initial_prov_cls = self._get_provider_class()
        initial_model_name = self.provider.model

        # Stable per-evaluation single-flight lock key based on inputs and template (router-resilient)
        lock_key = (
            self.cache._compute_hash(
                generated_answer=generated_answer,
                query=q,
                context=context,
                expected_answer=expected_answer,
                ground_truth=gt,
                provider_class="",
                model="",
                prompt_template=JUDGE_PROMPT_TEMPLATE,
                prompt_template_version="v2"
            )
            if self.cache
            else None
        )

        lock = await self.cache.get_lock_for_key(lock_key) if (self.cache and lock_key) else asyncio.Lock()

        async with lock:
            # Recompute/check cache inside single-flight lock across active or fallback router providers
            if self.cache:
                current_prov_cls = self._get_provider_class()
                current_model_name = self.provider.model
                cached = self.cache.get(
                    generated_answer=generated_answer,
                    query=q,
                    context=context,
                    expected_answer=expected_answer,
                    ground_truth=gt,
                    provider_class=current_prov_cls,
                    model=current_model_name,
                    prompt_template=JUDGE_PROMPT_TEMPLATE,
                    prompt_template_version="v2"
                )
                if cached is None and hasattr(self.provider, "providers"):
                    for p in self.provider.providers:
                        p_cls = getattr(p, "provider_name", p.__class__.__name__)
                        p_model = getattr(p, "model", "")
                        c = self.cache.get(
                            generated_answer=generated_answer,
                            query=q,
                            context=context,
                            expected_answer=expected_answer,
                            ground_truth=gt,
                            provider_class=p_cls,
                            model=p_model,
                            prompt_template=JUDGE_PROMPT_TEMPLATE,
                            prompt_template_version="v2"
                        )
                        if c is not None:
                            cached = c
                            break

                if cached is not None:
                    res = dict(cached)
                    res["passed"] = self._compute_passed(
                        res.get("faithfulness", 0.0),
                        res.get("answer_relevance", 0.0),
                        res.get("correctness", 0.0)
                    )
                    res["judge_mode"] = "cache"
                    res["judge_provenance"] = cached.get("judge_provenance", "unknown")
                    return res

            prompt = JUDGE_PROMPT_TEMPLATE.format(
                context=context,
                expected_answer=expected_answer,
                answer=generated_answer
            )

            input_tokens = 0
            output_tokens = 0
            judge_mode = "fallback"
            judge_provenance = "unknown"
            actual_prov_cls = initial_prov_cls
            actual_model_name = initial_model_name
            actual_execution_mode = "remote"

            if self._is_local_fallback_required():
                result_data = self._local_deterministic_grade(context, expected_answer, generated_answer)
                judge_mode = "fallback"
                judge_provenance = "deterministic_fallback"
                actual_execution_mode = "mock"
            else:
                try:
                    provider_resp = await self.provider.generate(prompt, json_mode=True)
                    if isinstance(provider_resp, ProviderResponse):
                        text_response = provider_resp.text
                        input_tokens = provider_resp.prompt_tokens
                        output_tokens = provider_resp.completion_tokens
                        if provider_resp.provider_name:
                            actual_prov_cls = provider_resp.provider_name
                        if provider_resp.model_name:
                            actual_model_name = provider_resp.model_name
                        actual_execution_mode = getattr(provider_resp, "execution_mode", "remote")
                    else:
                        text_response = str(provider_resp)

                    if actual_execution_mode == "local":
                        judge_mode = "fallback"
                        judge_provenance = "local_model"
                    elif actual_prov_cls in ["MockProvider", "MockRAGClient"] or actual_execution_mode == "mock":
                        judge_mode = "fallback"
                        judge_provenance = "mock"
                    elif actual_execution_mode == "remote":
                        judge_mode = "llm"
                        judge_provenance = "remote_llm"
                    else:
                        judge_mode = "fallback"
                        judge_provenance = "unknown"

                    cleaned_response = strip_markdown_json(text_response)
                    parsed_json = json.loads(cleaned_response)
                    validated = JudgeGenerationOutput.model_validate(parsed_json)
                    result_data = validated.model_dump()
                except Exception as e:
                    status_code = resolve_error_status_code(e)
                    msg = str(e).lower()
                    if status_code in [401, 403, 429, 500] or isinstance(e, (ValidationError, json.JSONDecodeError)) or "rate limit" in msg or "api key is required" in msg or "circuit open" in msg or "all providers in fallback chain failed" in msg:
                        result_data = self._local_deterministic_grade(context, expected_answer, generated_answer)
                        judge_mode = "fallback"
                        judge_provenance = "deterministic_fallback"
                        actual_execution_mode = "mock"
                    else:
                        raise

            faithfulness = result_data.get("faithfulness", 0.0)
            relevance = result_data.get("answer_relevance", 0.0)
            correctness = result_data.get("correctness", 0.0)
            passed = self._compute_passed(faithfulness, relevance, correctness)

            result = {
                **result_data,
                "passed": passed,
                "judge_mode": judge_mode,
                "judge_provenance": judge_provenance,
                "judge_prompt_tokens": input_tokens,
                "judge_completion_tokens": output_tokens,
            }

            if input_tokens == 0:
                input_tokens = max(1, len(prompt) // 4)
            if output_tokens == 0:
                output_tokens = max(1, len(json.dumps(result)) // 4)

            if judge_mode == "llm" and actual_execution_mode == "remote" and judge_provenance == "remote_llm":
                call_cost = calculate_token_cost(actual_model_name, input_tokens, output_tokens)
                async with self._cost_lock:
                    self.total_cost += call_cost
                    self.total_judge_prompt_tokens += input_tokens
                    self.total_judge_completion_tokens += output_tokens
                result["judge_cost"] = call_cost
            else:
                result["judge_cost"] = 0.0

            # Only cache authoritative remote judge evaluations with verified remote_llm provenance
            if self.cache and judge_mode == "llm" and actual_execution_mode == "remote" and judge_provenance == "remote_llm":
                cache_payload = {
                    "faithfulness": faithfulness,
                    "faithfulness_reasoning": result_data.get("faithfulness_reasoning", ""),
                    "answer_relevance": relevance,
                    "answer_relevance_reasoning": result_data.get("answer_relevance_reasoning", ""),
                    "correctness": correctness,
                    "correctness_reasoning": result_data.get("correctness_reasoning", ""),
                    "judge_mode": "llm",
                    "judge_provenance": "remote_llm",
                    "judge_prompt_tokens": input_tokens,
                    "judge_completion_tokens": output_tokens,
                    "judge_cost": result.get("judge_cost", 0.0)
                }
                # Write cache under the actual executing provider class and model name
                self.cache.set(
                    generated_answer=generated_answer,
                    query=q,
                    context=context,
                    expected_answer=expected_answer,
                    ground_truth=gt,
                    provider_class=actual_prov_cls,
                    model=actual_model_name,
                    prompt_template=JUDGE_PROMPT_TEMPLATE,
                    prompt_template_version="v2",
                    result=cache_payload
                )

            return result

    def _local_deterministic_grade(self, context: str, expected_answer: str, generated_answer: str) -> dict:
        """Deterministic fallback grading mechanism for local tests without API keys.
        Accurately calculates word overlap without artificial score floors.
        """
        gen_words = set(generated_answer.lower().split())
        exp_words = set(expected_answer.lower().split())

        gen_words = {w.strip(".,?!()\":;") for w in gen_words}
        exp_words = {w.strip(".,?!()\":;") for w in exp_words}

        stop_words = {"the", "a", "an", "is", "are", "of", "and", "in", "to", "for", "with", "on", "at", "by", "that", "this", "it"}
        gen_words -= stop_words
        exp_words -= stop_words

        if not exp_words:
            overlap = 1.0 if not gen_words else 0.5
        else:
            overlap = len(gen_words.intersection(exp_words)) / len(exp_words)

        score = round(min(1.0, max(0.0, overlap)), 2)

        return {
            "faithfulness": score,
            "faithfulness_reasoning": f"Fallback: Local word overlap score is {score:.2f}.",
            "answer_relevance": score,
            "answer_relevance_reasoning": f"Fallback: Local word overlap score is {score:.2f}.",
            "correctness": score,
            "correctness_reasoning": f"Fallback: Local word overlap score is {score:.2f}."
        }

    async def evaluate_retrieval(self, query: str, retrieved_contexts: Optional[List[str]], ground_truth: Optional[str]) -> dict:
        """Evaluates Context Precision and Context Recall for RAG retrieved context chunks with full token & cost accounting."""
        if not retrieved_contexts:
            return {
                "judge_mode": "fallback",
                "judge_provenance": "deterministic_fallback",
                "context_precision": 0.0,
                "context_precision_reasoning": "No retrieved contexts provided.",
                "context_recall": 0.0,
                "context_recall_reasoning": "No retrieved contexts provided.",
                "judge_prompt_tokens": 0,
                "judge_completion_tokens": 0,
                "judge_cost": 0.0
            }

        q = query or ""
        gt = ground_truth or ""
        formatted_contexts = "\n---\n".join(retrieved_contexts)
        initial_prov_cls = self._get_provider_class()
        initial_model_name = self.provider.model

        # Stable per-evaluation single-flight lock key based on inputs and template (router-resilient)
        lock_key = (
            self.cache._compute_hash(
                generated_answer=formatted_contexts,
                query=q,
                context=gt,
                expected_answer="",
                ground_truth=gt,
                provider_class="",
                model="",
                prompt_template=RETRIEVAL_JUDGE_PROMPT_TEMPLATE,
                prompt_template_version="v2"
            )
            if self.cache
            else None
        )

        lock = await self.cache.get_lock_for_key(lock_key) if (self.cache and lock_key) else asyncio.Lock()

        async with lock:
            if self.cache:
                current_prov_cls = self._get_provider_class()
                current_model_name = self.provider.model
                cached = self.cache.get(
                    generated_answer=formatted_contexts,
                    query=q,
                    context=gt,
                    expected_answer="",
                    ground_truth=gt,
                    provider_class=current_prov_cls,
                    model=current_model_name,
                    prompt_template=RETRIEVAL_JUDGE_PROMPT_TEMPLATE,
                    prompt_template_version="v2"
                )
                if cached is None and hasattr(self.provider, "providers"):
                    for p in self.provider.providers:
                        p_cls = getattr(p, "provider_name", p.__class__.__name__)
                        p_model = getattr(p, "model", "")
                        c = self.cache.get(
                            generated_answer=formatted_contexts,
                            query=q,
                            context=gt,
                            expected_answer="",
                            ground_truth=gt,
                            provider_class=p_cls,
                            model=p_model,
                            prompt_template=RETRIEVAL_JUDGE_PROMPT_TEMPLATE,
                            prompt_template_version="v2"
                        )
                        if c is not None:
                            cached = c
                            break

                if cached is not None:
                    cached_copy = dict(cached)
                    cached_copy["judge_mode"] = "cache"
                    cached_copy["judge_provenance"] = cached.get("judge_provenance", "unknown")
                    return cached_copy

            prompt = RETRIEVAL_JUDGE_PROMPT_TEMPLATE.format(
                query=q,
                ground_truth=gt,
                retrieved_contexts=formatted_contexts
            )

            input_tokens = 0
            output_tokens = 0
            judge_mode = "fallback"
            judge_provenance = "unknown"
            actual_prov_cls = initial_prov_cls
            actual_model_name = initial_model_name
            actual_execution_mode = "remote"

            if self._is_local_fallback_required():
                result_data = self._local_deterministic_retrieval_grade(q, retrieved_contexts, gt)
                judge_mode = "fallback"
                judge_provenance = "deterministic_fallback"
                actual_execution_mode = "mock"
            else:
                try:
                    provider_resp = await self.provider.generate(prompt, json_mode=True)
                    if isinstance(provider_resp, ProviderResponse):
                        text_response = provider_resp.text
                        input_tokens = provider_resp.prompt_tokens
                        output_tokens = provider_resp.completion_tokens
                        if provider_resp.provider_name:
                            actual_prov_cls = provider_resp.provider_name
                        if provider_resp.model_name:
                            actual_model_name = provider_resp.model_name
                        actual_execution_mode = getattr(provider_resp, "execution_mode", "remote")
                    else:
                        text_response = str(provider_resp)

                    if actual_execution_mode == "local":
                        judge_mode = "fallback"
                        judge_provenance = "local_model"
                    elif actual_prov_cls in ["MockProvider", "MockRAGClient"] or actual_execution_mode == "mock":
                        judge_mode = "fallback"
                        judge_provenance = "mock"
                    elif actual_execution_mode == "remote":
                        judge_mode = "llm"
                        judge_provenance = "remote_llm"
                    else:
                        judge_mode = "fallback"
                        judge_provenance = "unknown"

                    cleaned_response = strip_markdown_json(text_response)
                    parsed_json = json.loads(cleaned_response)
                    validated = JudgeRetrievalOutput.model_validate(parsed_json)
                    result_data = validated.model_dump()
                except Exception as e:
                    status_code = resolve_error_status_code(e)
                    msg = str(e).lower()
                    if status_code in [401, 403, 429, 500] or isinstance(e, (ValidationError, json.JSONDecodeError)) or "rate limit" in msg or "api key is required" in msg or "circuit open" in msg or "all providers in fallback chain failed" in msg:
                        result_data = self._local_deterministic_retrieval_grade(q, retrieved_contexts, gt)
                        judge_mode = "fallback"
                        judge_provenance = "deterministic_fallback"
                        actual_execution_mode = "mock"
                    else:
                        raise

            if input_tokens == 0:
                input_tokens = max(1, len(prompt) // 4)
            if output_tokens == 0:
                output_tokens = max(1, len(json.dumps(result_data)) // 4)

            call_cost = 0.0
            if judge_mode == "llm" and actual_execution_mode == "remote" and judge_provenance == "remote_llm":
                call_cost = calculate_token_cost(actual_model_name, input_tokens, output_tokens)
                async with self._cost_lock:
                    self.total_cost += call_cost
                    self.total_judge_prompt_tokens += input_tokens
                    self.total_judge_completion_tokens += output_tokens

            result = {
                **result_data,
                "judge_mode": judge_mode,
                "judge_provenance": judge_provenance,
                "judge_prompt_tokens": input_tokens,
                "judge_completion_tokens": output_tokens,
                "judge_cost": call_cost
            }

            if self.cache and judge_mode == "llm" and actual_execution_mode == "remote" and judge_provenance == "remote_llm":
                self.cache.set(
                    generated_answer=formatted_contexts,
                    query=q,
                    context=gt,
                    expected_answer="",
                    ground_truth=gt,
                    provider_class=actual_prov_cls,
                    model=actual_model_name,
                    prompt_template=RETRIEVAL_JUDGE_PROMPT_TEMPLATE,
                    prompt_template_version="v2",
                    result=result
                )

            return result

    def _local_deterministic_retrieval_grade(self, query: str, retrieved_contexts: Optional[List[str]], ground_truth: Optional[str]) -> dict:
        """Deterministic fallback retrieval grading for local execution without artificial score floors."""
        if not retrieved_contexts:
            return {
                "context_precision": 0.0,
                "context_precision_reasoning": "Local fallback: No retrieved contexts provided.",
                "context_recall": 0.0,
                "context_recall_reasoning": "Local fallback: No retrieved contexts provided."
            }

        gt = ground_truth or ""
        gt_words = {w.strip(".,?!()\":;").lower() for w in gt.split()} - {"the", "a", "an", "is", "are", "of", "and", "in", "to", "for", "with", "on", "at", "by", "that", "this", "it"}

        if not gt_words:
            return {
                "context_precision": 1.0,
                "context_precision_reasoning": "Local fallback: Empty ground truth.",
                "context_recall": 1.0,
                "context_recall_reasoning": "Local fallback: Empty ground truth."
            }

        combined_ctx = " ".join(retrieved_contexts).lower()
        ctx_words = {w.strip(".,?!()\":;") for w in combined_ctx.split()}

        matched = gt_words.intersection(ctx_words)
        recall = len(matched) / len(gt_words) if gt_words else 1.0
        precision = (len(matched) / len(ctx_words)) if ctx_words else 0.0

        p_val = round(min(1.0, max(0.0, precision * 2.0)), 2)
        r_val = round(min(1.0, max(0.0, recall)), 2)

        return {
            "context_precision": p_val,
            "context_precision_reasoning": f"Local fallback: Word overlap precision is {p_val}.",
            "context_recall": r_val,
            "context_recall_reasoning": f"Local fallback: Word overlap recall is {r_val}."
        }
