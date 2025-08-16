import json
import csv
from io import StringIO
from typing import List, Dict, Tuple, Any
import time
import asyncio
from datetime import datetime
from sqlalchemy.orm import Session

from app.models import Run, TestCase, MetricResult, AgentVersion, DatasetVersion

class AnswersProcessor:
    @staticmethod
    def process_answers_file(content: str, filename: str) -> Tuple[List[Dict], List[str]]:
        """Process uploaded answers file and return answers data and any errors"""
        errors = []
        answers = []
        
        try:
            # Determine file type
            if filename.endswith('.jsonl'):
                answers, file_errors = AnswersProcessor._process_jsonl(content)
                errors.extend(file_errors)
            elif filename.endswith('.csv'):
                answers, file_errors = AnswersProcessor._process_csv(content)
                errors.extend(file_errors)
            elif filename.endswith('.json'):
                answers, file_errors = AnswersProcessor._process_json(content)
                errors.extend(file_errors)
            else:
                errors.append(f"Unsupported file type. Please upload .jsonl, .csv, or .json file")
                return [], errors
                
        except Exception as e:
            errors.append(f"Error processing file: {str(e)}")
            return [], errors
            
        # Validate required fields
        validation_errors = AnswersProcessor._validate_answers(answers)
        errors.extend(validation_errors)
        
        return answers, errors
    
    @staticmethod
    def _process_jsonl(content: str) -> Tuple[List[Dict], List[str]]:
        """Process JSONL format answers file"""
        answers = []
        errors = []
        
        for line_num, line in enumerate(content.strip().split('\n'), 1):
            if not line.strip():
                continue
                
            try:
                answer = json.loads(line)
                answers.append(answer)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")
                
        return answers, errors
    
    @staticmethod
    def _process_csv(content: str) -> Tuple[List[Dict], List[str]]:
        """Process CSV format answers file"""
        answers = []
        errors = []
        
        try:
            reader = csv.DictReader(StringIO(content))
            for row_num, row in enumerate(reader, 1):
                if not any(row.values()):  # Skip empty rows
                    continue
                answers.append(dict(row))
        except Exception as e:
            errors.append(f"CSV parsing error: {str(e)}")
            
        return answers, errors
    
    @staticmethod
    def _process_json(content: str) -> Tuple[List[Dict], List[str]]:
        """Process JSON format answers file"""
        answers = []
        errors = []
        
        try:
            data = json.loads(content)
            if isinstance(data, list):
                answers = data
            else:
                errors.append("JSON file must contain an array of answer objects")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
            
        return answers, errors
    
    @staticmethod
    def _validate_answers(answers: List[Dict]) -> List[str]:
        """Validate that answers have required fields"""
        errors = []
        
        if not answers:
            errors.append("No valid answers found in file")
            return errors
            
        required_fields = ['id', 'answer']
        seen_ids = set()
        
        for i, answer in enumerate(answers, 1):
            # Check required fields
            for field in required_fields:
                if field not in answer:
                    errors.append(f"Answer {i}: Missing required field '{field}'")
                elif not answer[field] or str(answer[field]).strip() == "":
                    errors.append(f"Answer {i}: Field '{field}' cannot be empty")
            
            # Check for duplicate IDs
            answer_id = answer.get('id')
            if answer_id:
                if answer_id in seen_ids:
                    errors.append(f"Duplicate answer ID found: {answer_id}")
                seen_ids.add(answer_id)
                
        return errors
    
    @staticmethod
    def validate_coverage(answers: List[Dict], dataset_rows: List[Dict]) -> List[str]:
        """Validate that answers cover 100% of dataset cases"""
        errors = []
        
        answer_ids = {str(answer.get('id', '')) for answer in answers}
        dataset_ids = {str(row.get('id', '')) for row in dataset_rows}
        
        missing_ids = dataset_ids - answer_ids
        extra_ids = answer_ids - dataset_ids
        
        if missing_ids:
            errors.append(f"Missing answers for dataset IDs: {', '.join(sorted(missing_ids))}")
            
        if extra_ids:
            errors.append(f"Extra answer IDs not in dataset: {', '.join(sorted(extra_ids))}")
            
        return errors

class RunProcessor:
    """Handles the execution of evaluation runs"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def start_run(self, run: Run, answers: List[Dict], dataset_rows: List[Dict]) -> None:
        """Start processing a run with uploaded answers"""
        try:
            # Update run status
            run.status = "running"
            run.started_at = datetime.utcnow()
            run.total_test_cases = len(dataset_rows)
            self.db.commit()
            
            # Create test cases from dataset and answers
            answer_map = {str(answer['id']): answer['answer'] for answer in answers}
            
            for dataset_row in dataset_rows:
                case_id = str(dataset_row['id'])
                actual_answer = answer_map[case_id]
                
                test_case = TestCase(
                    run_id=run.id,
                    case_id=case_id,
                    question=dataset_row.get('question', ''),
                    context=dataset_row.get('context'),
                    expected_answer=dataset_row.get('expected_answer'),
                    actual_answer=actual_answer
                )
                self.db.add(test_case)
            
            self.db.commit()
            
            # Process each test case
            test_cases = self.db.query(TestCase).filter(TestCase.run_id == run.id).all()
            
            for test_case in test_cases:
                await self._process_test_case(test_case, run.agent_version)
                run.completed_test_cases += 1
                self.db.commit()
            
            # Calculate aggregates
            self._calculate_run_aggregates(run)
            
            # Mark run as completed
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            self.db.commit()
            
            # Generate export artifacts
            try:
                from app.export_utils import RunExporter
                exporter = RunExporter()
                
                # Generate all export formats
                csv_path = exporter.export_csv(run, self.db)
                json_path = exporter.export_json(run, self.db)
                html_path = exporter.generate_html_report(run, self.db)
                
                # Update run with artifact paths
                run.csv_export_path = csv_path
                run.json_export_path = json_path
                run.html_report_path = html_path
                self.db.commit()
                
            except Exception as e:
                # Don't fail the entire run if export fails
                print(f"Warning: Failed to generate export artifacts: {e}")
            
        except Exception as e:
            run.status = "failed"
            self.db.commit()
            raise e
    
    async def _process_test_case(self, test_case: TestCase, agent_version: AgentVersion) -> None:
        """Process a single test case with enabled metrics"""
        metric_scores = []
        
        # LLM-as-Judge scoring
        if agent_version.llm_as_judge_enabled:
            score, reasoning = await self._score_llm_as_judge(test_case, agent_version)
            threshold = agent_version.default_thresholds.get('llm_as_judge', 0.8)
            passed = score >= threshold
            
            metric_result = MetricResult(
                test_case_id=test_case.id,
                metric_name='llm_as_judge',
                score=score,
                passed=passed,
                threshold=threshold,
                reasoning=reasoning,
                execution_time_ms=100  # Mock execution time
            )
            self.db.add(metric_result)
            metric_scores.append(score)
        
        # Faithfulness scoring (only if context exists)
        if agent_version.faithfulness_enabled and test_case.context:
            score, reasoning = await self._score_faithfulness(test_case, agent_version)
            threshold = agent_version.default_thresholds.get('faithfulness', 0.8)
            passed = score >= threshold
            
            metric_result = MetricResult(
                test_case_id=test_case.id,
                metric_name='faithfulness',
                score=score,
                passed=passed,
                threshold=threshold,
                reasoning=reasoning,
                execution_time_ms=150  # Mock execution time
            )
            self.db.add(metric_result)
            metric_scores.append(score)
        
        # Calculate overall score for test case
        if metric_scores:
            test_case.overall_score = sum(metric_scores) / len(metric_scores)
            test_case.passed = all(
                result.passed for result in 
                self.db.query(MetricResult).filter(MetricResult.test_case_id == test_case.id).all()
            )
    
    async def _score_llm_as_judge(self, test_case: TestCase, agent_version: AgentVersion) -> Tuple[float, str]:
        """Score using LLM-as-Judge (simplified mock implementation)"""
        # This is a simplified mock implementation
        # In a real implementation, you would call the judge model API
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Mock scoring logic based on answer length and question match
        base_score = min(1.0, len(test_case.actual_answer) / 100.0 + 0.5)
        
        # Add some variance based on content
        if "error" in test_case.actual_answer.lower():
            score = max(0.1, base_score - 0.3)
            reasoning = "Answer contains error indicators, reducing confidence in response quality."
        elif len(test_case.actual_answer) < 10:
            score = max(0.2, base_score - 0.2)
            reasoning = "Answer is very brief, may lack sufficient detail."
        elif len(test_case.actual_answer) > 200:
            score = min(0.9, base_score + 0.1)
            reasoning = "Answer is comprehensive and detailed, showing good understanding."
        else:
            score = base_score
            reasoning = "Answer length and content appear appropriate for the question."
        
        return round(score, 2), reasoning
    
    async def _score_faithfulness(self, test_case: TestCase, agent_version: AgentVersion) -> Tuple[float, str]:
        """Score faithfulness to context (simplified mock implementation)"""
        # This is a simplified mock implementation
        # In a real implementation, you would use embedding similarity or other methods
        await asyncio.sleep(0.1)  # Simulate processing
        
        # Mock scoring based on presence of context keywords in answer
        if test_case.context and test_case.actual_answer:
            context_words = set(test_case.context.lower().split())
            answer_words = set(test_case.actual_answer.lower().split())
            overlap = len(context_words.intersection(answer_words))
            overlap_ratio = overlap / max(len(context_words), 1)
            
            score = min(1.0, overlap_ratio + 0.3)
            
            if overlap_ratio > 0.5:
                reasoning = f"Strong alignment with context. {overlap} key terms from context appear in answer."
            elif overlap_ratio > 0.2:
                reasoning = f"Moderate alignment with context. {overlap} terms match but answer could be more grounded."
            else:
                reasoning = f"Limited alignment with context. Only {overlap} terms match, answer may not be well-grounded."
            
            return round(score, 2), reasoning
        
        return 0.5, "No context provided for faithfulness evaluation."
    
    def _calculate_run_aggregates(self, run: Run) -> None:
        """Calculate aggregate metrics for the run"""
        test_cases = self.db.query(TestCase).filter(TestCase.run_id == run.id).all()
        
        if not test_cases:
            return
        
        # Overall score (average of all test case scores)
        scores = [tc.overall_score for tc in test_cases if tc.overall_score is not None]
        if scores:
            run.overall_score = sum(scores) / len(scores)
        
        # Pass rate
        passed_cases = sum(1 for tc in test_cases if tc.passed)
        run.pass_rate = passed_cases / len(test_cases)
        
        # Aggregate results by metric
        aggregate_results = {}
        
        # Get all metric results
        all_metrics = self.db.query(MetricResult).join(TestCase).filter(TestCase.run_id == run.id).all()
        
        # Group by metric name
        metrics_by_name = {}
        for metric in all_metrics:
            if metric.metric_name not in metrics_by_name:
                metrics_by_name[metric.metric_name] = []
            metrics_by_name[metric.metric_name].append(metric)
        
        # Calculate aggregates for each metric
        for metric_name, metric_results in metrics_by_name.items():
            scores = [m.score for m in metric_results]
            passed_count = sum(1 for m in metric_results if m.passed)
            
            aggregate_results[metric_name] = {
                'score': sum(scores) / len(scores) if scores else 0,
                'pass_rate': passed_count / len(metric_results) if metric_results else 0,
                'total_cases': len(metric_results)
            }
        
        run.aggregate_results = aggregate_results