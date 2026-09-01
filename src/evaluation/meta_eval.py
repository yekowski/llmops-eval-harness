import asyncio
from typing import List, Dict, Tuple
from src.evaluation.judge import LLMJudge

def calculate_agreement_metrics(y_true: List[bool], y_pred: List[bool]) -> Tuple[float, float]:
    """Calculates observed agreement rate (accuracy) and Cohen's Kappa score."""
    total = len(y_true)
    if total == 0:
        return 0.0, 0.0
    
    # Observed agreement (accuracy)
    agreed = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = agreed / total
    
    # Expected agreement by chance
    p_true_actual = sum(1 for t in y_true if t) / total
    p_false_actual = 1.0 - p_true_actual
    
    p_true_pred = sum(1 for p in y_pred if p) / total
    p_false_pred = 1.0 - p_true_pred
    
    pe = (p_true_actual * p_true_pred) + (p_false_actual * p_false_pred)
    
    # Guard against division by zero if there's complete chance agreement/no variance
    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (accuracy - pe) / (1.0 - pe)
        
    return accuracy, kappa

async def run_meta_evaluation(dataset: List[Dict], judge: LLMJudge) -> Dict:
    """Runs LLM Judge evaluations concurrently over the dataset and computes meta-eval metrics."""
    tasks = []
    for item in dataset:
        tasks.append(judge.evaluate(
            context=item["context"],
            expected_answer=item["expected_answer"],
            generated_answer=item["generated_answer"],
            query=item.get("query")
        ))
        
    results = await asyncio.gather(*tasks)
    
    y_true = [item["expected_pass_boolean"] for item in dataset]
    y_pred = [res["passed"] for res in results]
    
    accuracy, kappa = calculate_agreement_metrics(y_true, y_pred)
    
    details = []
    for item, pred_res in zip(dataset, results):
        agreed = item["expected_pass_boolean"] == pred_res["passed"]
        details.append({
            "query": item["query"],
            "expected_pass_boolean": item["expected_pass_boolean"],
            "judge_passed": pred_res["passed"],
            "judge_explanation": pred_res.get("explanation", ""),
            "agreed": agreed
        })
        
    summary = {
        "accuracy": accuracy,
        "cohens_kappa": kappa,
        "details": details,
        "raw_results": results
    }
    
    if accuracy < 0.85:
        print(f"\n[WARNING] LLM Judge drift detected! Agreement rate is {accuracy:.2%} (Threshold: 85.00%).")
        
    return summary
