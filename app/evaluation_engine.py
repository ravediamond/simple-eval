"""
Real evaluation engine inspired by DeepEval concepts but using our existing connector infrastructure.
Provides LLM-as-Judge and Faithfulness metrics with proper scoring and explanations.
"""

import asyncio
import json
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from app.connectors import ConnectorFactory, ConnectorConfig, ConnectorResponse

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

class BaseMetric(ABC):
    """Abstract base class for evaluation metrics"""
    
    def __init__(self, threshold: float = 0.8, judge_config: Optional[ConnectorConfig] = None):
        self.threshold = threshold
        self.judge_config = judge_config
        
    @abstractmethod
    async def evaluate(
        self, 
        question: str, 
        actual_answer: str, 
        expected_answer: Optional[str] = None,
        context: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate the given inputs and return a result"""
        pass
    
    def _extract_score_from_response(self, response: str) -> Tuple[float, str]:
        """Extract binary result and reasoning from LLM response"""
        # First check for new PASS/FAIL format
        pass_fail_match = re.search(r"result:\s*(pass|fail)", response.lower())
        if pass_fail_match:
            result = pass_fail_match.group(1).lower()
            score = 1.0 if result == "pass" else 0.0
            
            # Extract explanation section
            explanation_match = re.search(r"explanation:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
            if explanation_match:
                reasoning = explanation_match.group(1).strip()
            else:
                reasoning = f"Result: {result.upper()}"
            
            return score, reasoning
        
        # Fallback to legacy score patterns for backward compatibility
        score_patterns = [
            r"(?:score|rating):\s*([0-9]*\.?[0-9]+)",
            r"([0-9]*\.?[0-9]+)\s*%",  # Percentage pattern
            r"([0-9]*\.?[0-9]+)\s*/\s*10",
            r"([0-9]*\.?[0-9]+)\s*/\s*5",
            r"([0-9]*\.?[0-9]+)\s*out of\s*(?:10|5)",
            r"(?:^|\s)([0-9]*\.?[0-9]+)(?:\s|$)",  # Any number
        ]
        
        score = None
        for i, pattern in enumerate(score_patterns):
            match = re.search(pattern, response.lower())
            if match:
                try:
                    raw_score = float(match.group(1))
                    # Normalize to 0-1 based on pattern type and scale
                    if i == 1:  # Percentage pattern (index 1)
                        score = raw_score / 100.0
                    elif raw_score <= 1.0:
                        score = raw_score
                    elif raw_score <= 5.0:
                        score = raw_score / 5.0
                    elif raw_score <= 10.0:
                        score = raw_score / 10.0
                    else:
                        score = min(1.0, raw_score / 100.0)  # Assume percentage
                    break
                except ValueError:
                    continue
        
        # If no score found, try to infer from sentiment
        if score is None:
            response_lower = response.lower()
            if any(word in response_lower for word in ['excellent', 'perfect', 'outstanding', 'pass']):
                score = 1.0
            elif any(word in response_lower for word in ['good', 'well', 'accurate', 'correct']):
                score = 0.8
            elif any(word in response_lower for word in ['fair', 'adequate', 'okay']):
                score = 0.6
            elif any(word in response_lower for word in ['poor', 'incorrect', 'wrong', 'bad', 'fail']):
                score = 0.0
            else:
                score = 0.5  # Default neutral score
        
        # Extract reasoning (everything that's not the score)
        reasoning = response.strip()
        for pattern in score_patterns:
            reasoning = re.sub(pattern, '', reasoning, flags=re.IGNORECASE)
        
        reasoning = re.sub(r'\s+', ' ', reasoning).strip()
        if not reasoning:
            reasoning = f"Assigned score: {score:.2f}"
            
        return max(0.0, min(1.0, score)), reasoning

class LLMAsJudgeMetric(BaseMetric):
    """LLM-as-Judge correctness evaluation metric"""
    
    def __init__(
        self, 
        threshold: float = 0.8, 
        judge_config: Optional[ConnectorConfig] = None,
        custom_prompt: Optional[str] = None,
        judge_profile: str = "simple"
    ):
        super().__init__(threshold, judge_config)
        self.judge_profile = judge_profile
        self.custom_prompt = custom_prompt or self._get_profile_prompt(judge_profile)
    
    def _get_profile_prompt(self, profile: str) -> str:
        """Get the predefined prompt for a judge profile"""
        if profile == "simple":
            return self._get_simple_prompt()
        elif profile == "detailed":
            return self._get_detailed_prompt()
        else:  # custom or fallback
            return self._get_default_prompt()
    
    def _get_simple_prompt(self) -> str:
        return """You are an expert evaluator. Your task is to determine if the AI response correctly answers the question.

Question: {question}
{expected_section}
Response to evaluate: {actual_answer}

Evaluate ONLY if the response is correct. Respond with:
- PASS if the answer is correct
- FAIL if the answer is incorrect

Format your response as:
Result: [PASS/FAIL]
Explanation:
• [Bullet point explaining why it passes or fails]
• [Additional bullet points as needed]"""
    
    def _get_detailed_prompt(self) -> str:
        return """You are an expert evaluator. Evaluate the AI response against the expected answer with these strict criteria:

1. CORRECTNESS: The answer must be factually correct
2. COMPLETENESS: All facts from the expected answer must be present
3. NO EXTRA FACTS: The response should not contain facts not in the expected answer
4. ANSWER TYPE MATCH: The response style must match (detailed vs concise)

Question: {question}
{expected_section}
Response to evaluate: {actual_answer}

The response must meet ALL criteria to pass. Respond with:
- PASS if ALL criteria are met
- FAIL if ANY criteria fails

Format your response as:
Result: [PASS/FAIL]
Explanation:
• Correctness: [Pass/Fail - explain]
• Completeness: [Pass/Fail - explain] 
• No extra facts: [Pass/Fail - explain]
• Answer type match: [Pass/Fail - explain]"""
    
    def _get_default_prompt(self) -> str:
        return """You are an expert evaluator assessing the quality and correctness of AI responses.

Please evaluate the given response to the question on the following criteria:
1. Accuracy: Is the response factually correct?
2. Completeness: Does it fully answer the question?
3. Clarity: Is it well-written and understandable?
4. Relevance: Does it directly address what was asked?

Question: {question}
{expected_section}
Response to evaluate: {actual_answer}

Respond with:
- PASS if the response meets the criteria
- FAIL if the response does not meet the criteria

Format your response as:
Result: [PASS/FAIL]
Explanation:
• [Bullet point explaining your evaluation]
• [Additional bullet points as needed]"""
    
    async def evaluate(
        self, 
        question: str, 
        actual_answer: str, 
        expected_answer: Optional[str] = None,
        context: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate using LLM-as-Judge"""
        
        # Prepare prompt
        expected_section = ""
        if expected_answer:
            expected_section = f"Expected answer: {expected_answer}\n"
        
        prompt = self.custom_prompt.format(
            question=question,
            actual_answer=actual_answer,
            expected_section=expected_section
        ).strip()
        
        # Use judge connector to evaluate
        if not self.judge_config:
            raise ValueError("Judge connector configuration required for LLM-as-Judge evaluation")
        
        connector = ConnectorFactory.create_connector(self.judge_config)
        
        async with connector:
            response = await connector.evaluate(prompt)
            
            if not response.success:
                return EvaluationResult(
                    score=0.0,
                    reasoning=f"Evaluation failed: {response.error_message}",
                    passed=False,
                    threshold=self.threshold,
                    execution_time_ms=response.response_time_ms
                )
            
            # Extract score and reasoning
            score, reasoning = self._extract_score_from_response(response.answer)
            
            return EvaluationResult(
                score=score,
                reasoning=reasoning,
                passed=score >= self.threshold,
                threshold=self.threshold,
                raw_response=response.answer,
                prompt_used=prompt,
                execution_time_ms=response.response_time_ms
            )

class FaithfulnessMetric(BaseMetric):
    """Faithfulness to context evaluation metric"""
    
    def __init__(
        self, 
        threshold: float = 0.8, 
        judge_config: Optional[ConnectorConfig] = None
    ):
        super().__init__(threshold, judge_config)
    
    def _get_faithfulness_prompt(self) -> str:
        return """You are an expert evaluator assessing whether a response is faithful to the given context.

Faithfulness means the response:
1. Only uses information that can be found in the context
2. Does not contradict the context
3. Does not hallucinate or add unsupported claims
4. Accurately represents the information from the context

Please evaluate how faithful the response is to the provided context.

Context: {context}
Question: {question}
Response to evaluate: {actual_answer}

Provide a faithfulness score from 0 to 1:
- 1.0: Completely faithful, all claims are supported by context
- 0.8: Mostly faithful, minor unsupported details
- 0.6: Somewhat faithful, some unsupported claims
- 0.4: Partially faithful, significant unsupported content
- 0.2: Mostly unfaithful, contradicts or ignores context
- 0.0: Completely unfaithful, entirely unsupported

Please provide your evaluation in this format:
Score: [0.0-1.0]
Reasoning: [Explain which parts are supported/unsupported by context]"""
    
    async def evaluate(
        self, 
        question: str, 
        actual_answer: str, 
        expected_answer: Optional[str] = None,
        context: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate faithfulness to context"""
        
        if not context:
            return EvaluationResult(
                score=1.0,  # No context to be unfaithful to
                reasoning="No context provided - faithfulness not applicable",
                passed=True,
                threshold=self.threshold
            )
        
        # Prepare prompt
        prompt = self._get_faithfulness_prompt().format(
            context=context,
            question=question,
            actual_answer=actual_answer
        ).strip()
        
        # Use judge connector to evaluate
        if not self.judge_config:
            raise ValueError("Judge connector configuration required for faithfulness evaluation")
        
        connector = ConnectorFactory.create_connector(self.judge_config)
        
        async with connector:
            response = await connector.evaluate(prompt)
            
            if not response.success:
                return EvaluationResult(
                    score=0.0,
                    reasoning=f"Evaluation failed: {response.error_message}",
                    passed=False,
                    threshold=self.threshold,
                    execution_time_ms=response.response_time_ms
                )
            
            # Extract score and reasoning
            score, reasoning = self._extract_score_from_response(response.answer)
            
            return EvaluationResult(
                score=score,
                reasoning=reasoning,
                passed=score >= self.threshold,
                threshold=self.threshold,
                raw_response=response.answer,
                prompt_used=prompt,
                execution_time_ms=response.response_time_ms
            )

class EvaluationEngine:
    """Main evaluation engine that coordinates metric evaluation"""
    
    def __init__(self, judge_config: ConnectorConfig):
        self.judge_config = judge_config
    
    async def evaluate_test_case(
        self,
        question: str,
        actual_answer: str,
        expected_answer: Optional[str] = None,
        context: Optional[str] = None,
        llm_as_judge_threshold: float = 0.8,
        custom_judge_prompt: Optional[str] = None,
        judge_profile: str = "simple",
        store_verbose_artifacts: bool = False
    ) -> Dict[str, EvaluationResult]:
        """Evaluate a test case with LLM-as-Judge using the specified profile"""
        
        results = {}
        
        # LLM-as-Judge evaluation with profile support
        llm_judge = LLMAsJudgeMetric(
            threshold=llm_as_judge_threshold,
            judge_config=self.judge_config,
            custom_prompt=custom_judge_prompt,
            judge_profile=judge_profile
        )
        results['llm_as_judge'] = await llm_judge.evaluate(
            question, actual_answer, expected_answer, context
        )
        
        # Clean up verbose artifacts if not requested
        if not store_verbose_artifacts:
            for result in results.values():
                result.raw_response = None
                result.prompt_used = None
        
        return results
    
    def calculate_overall_score(self, metric_results: Dict[str, EvaluationResult]) -> Tuple[float, bool]:
        """Calculate overall score and pass status from individual metrics"""
        if not metric_results:
            return 0.0, False
        
        scores = [result.score for result in metric_results.values()]
        overall_score = sum(scores) / len(scores)
        
        # Overall pass requires all metrics to pass
        overall_passed = all(result.passed for result in metric_results.values())
        
        return overall_score, overall_passed