import json
import csv
from io import StringIO
from typing import List, Dict, Tuple, Any
import time
import asyncio
from datetime import datetime
from sqlalchemy.orm import Session

from app.models import Run, TestCase, MetricResult, AgentVersion, DatasetVersion, LLMConfiguration
from app.connectors import ConnectorFactory, ConnectorConfig, ConnectorManager
from app.evaluation_engine import EvaluationEngine

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
    
    async def start_run(self, run: Run, answers: List[Dict] = None, dataset_rows: List[Dict] = None) -> None:
        """Start processing a run with uploaded answers or live evaluation"""
        try:
            # Update run status
            run.status = "running"
            run.started_at = datetime.utcnow()
            
            if run.evaluation_source == "upload":
                # Handle uploaded answers workflow
                if not answers or not dataset_rows:
                    raise ValueError("Answers and dataset rows required for upload evaluation")
                
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
            
            elif run.evaluation_source == "connector":
                # Handle live evaluation workflow
                await self._start_live_evaluation(run, dataset_rows)
            
            else:
                raise ValueError(f"Unknown evaluation source: {run.evaluation_source}")
            
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
    
    async def _start_live_evaluation(self, run: Run, dataset_rows: List[Dict]) -> None:
        """Start live evaluation using connector"""
        if not dataset_rows:
            raise ValueError("Dataset rows required for live evaluation")
        
        if not run.agent_version.connector_enabled or not run.agent_version.connector_config:
            raise ValueError("Connector not configured for this agent version")
        
        run.total_test_cases = len(dataset_rows)
        self.db.commit()
        
        # Create connector from config
        connector_config = ConnectorConfig(**run.agent_version.connector_config)
        connector = ConnectorFactory.create_connector(connector_config)
        connector_manager = ConnectorManager(max_concurrent=5)
        
        async with connector:
            # Prepare questions for batch evaluation
            questions = []
            for dataset_row in dataset_rows:
                questions.append({
                    "id": str(dataset_row['id']),
                    "question": dataset_row.get('question', ''),
                    "context": dataset_row.get('context')
                })
            
            # Evaluate in batches
            batch_size = 10
            for i in range(0, len(questions), batch_size):
                batch = questions[i:i + batch_size]
                
                # Get responses from connector
                responses = await connector_manager.evaluate_batch(connector, batch)
                
                # Create test cases and process them
                for j, response in enumerate(responses):
                    dataset_row = dataset_rows[i + j]
                    case_id = str(dataset_row['id'])
                    
                    # Create test case with connector response
                    test_case = TestCase(
                        run_id=run.id,
                        case_id=case_id,
                        question=dataset_row.get('question', ''),
                        context=dataset_row.get('context'),
                        expected_answer=dataset_row.get('expected_answer'),
                        actual_answer=response.answer if response.success else f"Error: {response.error_message}"
                    )
                    self.db.add(test_case)
                    self.db.flush()  # Get the ID
                    
                    # Process test case for scoring
                    await self._process_test_case(test_case, run.agent_version)
                    
                    # Store connector metadata if available
                    if response.metadata:
                        # Could store in a separate table or in test case notes
                        pass
                    
                    run.completed_test_cases += 1
                    self.db.commit()
    
    async def _process_test_case(self, test_case: TestCase, agent_version: AgentVersion) -> None:
        """Process a single test case with enabled metrics using real evaluation engine"""
        
        # Try to get default judge configuration from database first
        default_judge_config = self.db.query(LLMConfiguration).filter(
            LLMConfiguration.is_default_judge == True,
            LLMConfiguration.is_active == True
        ).first()
        
        if default_judge_config:
            # Use saved LLM configuration as judge
            judge_config = default_judge_config.to_connector_config()
            # Override judge prompt if agent has custom prompt
            custom_prompt = agent_version.judge_prompt or default_judge_config.default_judge_prompt
            llm_as_judge_threshold = default_judge_config.llm_as_judge_threshold
            faithfulness_threshold = default_judge_config.faithfulness_threshold
        else:
            # Fallback to agent version judge config
            judge_config = ConnectorConfig(**agent_version.judge_model_config)
            custom_prompt = agent_version.judge_prompt
            llm_as_judge_threshold = agent_version.default_thresholds.get('llm_as_judge', 0.8)
            faithfulness_threshold = agent_version.default_thresholds.get('faithfulness', 0.8)
        
        evaluation_engine = EvaluationEngine(judge_config)
        
        try:
            # Evaluate test case with real metrics
            metric_results = await evaluation_engine.evaluate_test_case(
                question=test_case.question,
                actual_answer=test_case.actual_answer,
                expected_answer=test_case.expected_answer,
                context=test_case.context,
                llm_as_judge_enabled=agent_version.llm_as_judge_enabled,
                faithfulness_enabled=agent_version.faithfulness_enabled,
                llm_as_judge_threshold=llm_as_judge_threshold,
                faithfulness_threshold=faithfulness_threshold,
                custom_judge_prompt=custom_prompt,
                store_verbose_artifacts=agent_version.store_verbose_artifacts or False
            )
            
            # Store metric results in database
            metric_scores = []
            for metric_name, eval_result in metric_results.items():
                metric_result = MetricResult(
                    test_case_id=test_case.id,
                    metric_name=metric_name,
                    score=eval_result.score,
                    passed=eval_result.passed,
                    threshold=eval_result.threshold,
                    reasoning=eval_result.reasoning,
                    execution_time_ms=eval_result.execution_time_ms,
                    raw_judge_response=eval_result.raw_response,
                    judge_prompt_used=eval_result.prompt_used
                )
                self.db.add(metric_result)
                metric_scores.append(eval_result.score)
            
            # Calculate overall score and pass status
            if metric_scores:
                overall_score, overall_passed = evaluation_engine.calculate_overall_score(metric_results)
                test_case.overall_score = overall_score
                test_case.passed = overall_passed
            else:
                test_case.overall_score = 0.0
                test_case.passed = False
                
        except Exception as e:
            # Handle evaluation failures gracefully
            print(f"Evaluation failed for test case {test_case.case_id}: {e}")
            
            # Create error metric result
            error_metric = MetricResult(
                test_case_id=test_case.id,
                metric_name='evaluation_error',
                score=0.0,
                passed=False,
                threshold=0.8,
                reasoning=f"Evaluation failed: {str(e)}",
                execution_time_ms=0
            )
            self.db.add(error_metric)
            
            test_case.overall_score = 0.0
            test_case.passed = False
    
    
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