"""
Real DeepEval integration for evaluation metrics.
This replaces the custom evaluation engine with the actual DeepEval library.
"""

import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from deepeval import evaluate
from deepeval.models import OpenAIGPT
from deepeval.metrics import GEval, FaithfulnessMetric as DeepEvalFaithfulness
from deepeval.test_case import LLMTestCase

from app.connectors import ConnectorConfig

@dataclass
class EvaluationResult:
    """Result from a metric evaluation"""
    score: float  # Normalized 0-1 score
    reasoning: str  # Explanation of the score
    passed: bool  # Whether it meets the threshold
    threshold: float  # Threshold used
    raw_response: Optional[str] = None  # Raw judge response
    prompt_used: Optional[str] = None  # Prompt sent to judge
    execution_time_ms: Optional[int] = None

class DeepEvalEngine:
    """Evaluation engine using DeepEval library"""
    
    def __init__(self, judge_config: ConnectorConfig):
        self.judge_config = judge_config
        self._model = None
    
    def _get_model(self):
        """Get DeepEval model instance"""
        if self._model is None:
            if self.judge_config.connector_type == "openai":
                # Use OpenAI model for DeepEval
                self._model = OpenAIGPT(
                    model=self.judge_config.model_name,
                    api_key=self.judge_config.api_key,
                    temperature=self.judge_config.temperature
                )
            else:
                # For non-OpenAI connectors, we'll use OpenAI as the judge model
                # but get API key from environment or config
                import os
                api_key = os.getenv("OPENAI_API_KEY") or self.judge_config.api_key
                self._model = OpenAIGPT(
                    model="gpt-4",
                    api_key=api_key,
                    temperature=0.0
                )
        return self._model
    
    async def evaluate_test_case(
        self,
        question: str,
        actual_answer: str,
        expected_answer: Optional[str] = None,
        context: Optional[str] = None,
        llm_as_judge_enabled: bool = True,
        faithfulness_enabled: bool = True,
        llm_as_judge_threshold: float = 0.8,
        faithfulness_threshold: float = 0.8,
        custom_judge_prompt: Optional[str] = None,
        store_verbose_artifacts: bool = False
    ) -> Dict[str, EvaluationResult]:
        """Evaluate a test case with DeepEval metrics"""
        
        results = {}
        
        # Create DeepEval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer,
            context=context
        )
        
        metrics = []
        
        # Add LLM-as-Judge metric (using GEval)
        if llm_as_judge_enabled:
            judge_prompt = custom_judge_prompt or """
You are an expert evaluator assessing the quality and correctness of AI responses.

Please evaluate the given response on the following criteria:
1. Accuracy: Is the response factually correct?
2. Completeness: Does it fully answer the question?
3. Clarity: Is it well-written and understandable?
4. Relevance: Does it directly address what was asked?

Please be strict in your evaluation.
            """.strip()
            
            llm_judge_metric = GEval(
                name="llm_as_judge",
                criteria=judge_prompt,
                evaluation_params=["accuracy", "completeness", "clarity", "relevance"],
                model=self._get_model(),
                threshold=llm_as_judge_threshold
            )
            metrics.append(llm_judge_metric)
        
        # Add Faithfulness metric
        if faithfulness_enabled and context:
            faithfulness_metric = DeepEvalFaithfulness(
                model=self._get_model(),
                threshold=faithfulness_threshold
            )
            metrics.append(faithfulness_metric)
        
        # Run evaluation
        start_time = time.time()
        
        try:
            # Run DeepEval evaluation
            evaluation_results = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._run_evaluation, 
                test_case, 
                metrics
            )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Process results
            for metric in metrics:
                metric_name = metric.name if hasattr(metric, 'name') else type(metric).__name__.lower()
                
                # Get score and reason from metric
                score = getattr(metric, 'score', 0.0)
                reason = getattr(metric, 'reason', 'No reasoning provided')
                success = getattr(metric, 'success', False)
                threshold = getattr(metric, 'threshold', 0.8)
                
                # Normalize score to 0-1 range
                if score > 1.0:
                    score = score / 10.0 if score <= 10.0 else score / 100.0
                
                # Get verbose artifacts if requested
                raw_response = None
                prompt_used = None
                if store_verbose_artifacts:
                    raw_response = getattr(metric, 'evaluation_model_response', None)
                    prompt_used = getattr(metric, 'evaluation_prompt', None)
                
                results[metric_name] = EvaluationResult(
                    score=max(0.0, min(1.0, score)),
                    reasoning=reason,
                    passed=success,
                    threshold=threshold,
                    raw_response=raw_response,
                    prompt_used=prompt_used,
                    execution_time_ms=execution_time_ms
                )
            
            return results
            
        except Exception as e:
            # Handle evaluation failures
            error_msg = f"DeepEval evaluation failed: {str(e)}"
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Return error results for enabled metrics
            if llm_as_judge_enabled:
                results['llm_as_judge'] = EvaluationResult(
                    score=0.0,
                    reasoning=error_msg,
                    passed=False,
                    threshold=llm_as_judge_threshold,
                    execution_time_ms=execution_time_ms
                )
            
            if faithfulness_enabled and context:
                results['faithfulness'] = EvaluationResult(
                    score=0.0,
                    reasoning=error_msg,
                    passed=False,
                    threshold=faithfulness_threshold,
                    execution_time_ms=execution_time_ms
                )
            
            return results
    
    def _run_evaluation(self, test_case, metrics):
        """Run DeepEval evaluation synchronously"""
        return evaluate(test_cases=[test_case], metrics=metrics)
    
    def calculate_overall_score(self, metric_results: Dict[str, EvaluationResult]) -> Tuple[float, bool]:
        """Calculate overall score and pass status from individual metrics"""
        if not metric_results:
            return 0.0, False
        
        scores = [result.score for result in metric_results.values()]
        overall_score = sum(scores) / len(scores)
        
        # Overall pass requires all metrics to pass
        overall_passed = all(result.passed for result in metric_results.values())
        
        return overall_score, overall_passed