import os
import csv
import json
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from jinja2 import Template

from app.models import Run, TestCase, MetricResult

class RunExporter:
    """Handles exporting run results to various formats"""
    
    def __init__(self, export_dir: str = "data/exports"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
    
    def export_csv(self, run: Run, db: Session) -> str:
        """Export run results to CSV format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{run.id}_{timestamp}.csv"
        filepath = os.path.join(self.export_dir, filename)
        
        test_cases = db.query(TestCase).filter(TestCase.run_id == run.id).order_by(TestCase.case_id).all()
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'case_id', 'question', 'actual_answer', 'expected_answer', 'context',
                'overall_score', 'passed', 'llm_as_judge_score', 'llm_as_judge_passed',
                'faithfulness_score', 'faithfulness_passed'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for test_case in test_cases:
                # Get metric results
                metric_results = {mr.metric_name: mr for mr in test_case.metric_results}
                
                row = {
                    'case_id': test_case.case_id,
                    'question': test_case.question,
                    'actual_answer': test_case.actual_answer,
                    'expected_answer': test_case.expected_answer or '',
                    'context': test_case.context or '',
                    'overall_score': test_case.overall_score,
                    'passed': test_case.passed,
                    'llm_as_judge_score': metric_results.get('llm_as_judge', {}).score if 'llm_as_judge' in metric_results else '',
                    'llm_as_judge_passed': metric_results.get('llm_as_judge', {}).passed if 'llm_as_judge' in metric_results else '',
                    'faithfulness_score': metric_results.get('faithfulness', {}).score if 'faithfulness' in metric_results else '',
                    'faithfulness_passed': metric_results.get('faithfulness', {}).passed if 'faithfulness' in metric_results else ''
                }
                writer.writerow(row)
        
        return filepath
    
    def export_json(self, run: Run, db: Session) -> str:
        """Export run results to JSON format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{run.id}_{timestamp}.json"
        filepath = os.path.join(self.export_dir, filename)
        
        test_cases = db.query(TestCase).filter(TestCase.run_id == run.id).order_by(TestCase.case_id).all()
        
        export_data = {
            'run_info': {
                'id': run.id,
                'name': run.name,
                'agent': run.agent_version.agent.name,
                'agent_version': run.agent_version.display_name,
                'dataset': run.dataset_version.dataset.name,
                'dataset_version': run.dataset_version.display_name,
                'status': run.status,
                'created_at': run.created_at.isoformat(),
                'started_at': run.started_at.isoformat() if run.started_at else None,
                'completed_at': run.completed_at.isoformat() if run.completed_at else None,
                'total_test_cases': run.total_test_cases,
                'overall_score': run.overall_score,
                'pass_rate': run.pass_rate,
                'aggregate_results': run.aggregate_results
            },
            'test_cases': []
        }
        
        for test_case in test_cases:
            case_data = {
                'case_id': test_case.case_id,
                'question': test_case.question,
                'actual_answer': test_case.actual_answer,
                'expected_answer': test_case.expected_answer,
                'context': test_case.context,
                'overall_score': test_case.overall_score,
                'passed': test_case.passed,
                'metrics': {}
            }
            
            for metric_result in test_case.metric_results:
                case_data['metrics'][metric_result.metric_name] = {
                    'score': metric_result.score,
                    'passed': metric_result.passed,
                    'threshold': metric_result.threshold,
                    'reasoning': metric_result.reasoning,
                    'execution_time_ms': metric_result.execution_time_ms
                }
            
            export_data['test_cases'].append(case_data)
        
        with open(filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)
        
        return filepath
    
    def generate_html_report(self, run: Run, db: Session) -> str:
        """Generate static HTML report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{run.id}_report_{timestamp}.html"
        filepath = os.path.join(self.export_dir, filename)
        
        test_cases = db.query(TestCase).filter(TestCase.run_id == run.id).order_by(TestCase.case_id).all()
        
        # Prepare data for template
        template_data = {
            'run': run,
            'test_cases': test_cases,
            'generated_at': datetime.now(),
            'total_cases': len(test_cases),
            'passed_cases': sum(1 for tc in test_cases if tc.passed),
            'failed_cases': sum(1 for tc in test_cases if not tc.passed),
        }
        
        # Calculate metric summaries
        metric_summaries = {}
        all_metrics = db.query(MetricResult).join(TestCase).filter(TestCase.run_id == run.id).all()
        
        for metric in all_metrics:
            if metric.metric_name not in metric_summaries:
                metric_summaries[metric.metric_name] = {
                    'scores': [],
                    'passed_count': 0,
                    'total_count': 0
                }
            
            metric_summaries[metric.metric_name]['scores'].append(metric.score)
            metric_summaries[metric.metric_name]['total_count'] += 1
            if metric.passed:
                metric_summaries[metric.metric_name]['passed_count'] += 1
        
        template_data['metric_summaries'] = metric_summaries
        
        # HTML template
        html_template = Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ run.name }} - Evaluation Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; line-height: 1.6; }
        .header { border-bottom: 2px solid #dee2e6; padding-bottom: 20px; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; text-align: center; }
        .summary-card h3 { margin: 0 0 10px 0; font-size: 2em; }
        .summary-card .label { color: #6c757d; font-size: 0.9em; }
        .success { color: #198754; }
        .warning { color: #fd7e14; }
        .danger { color: #dc3545; }
        .info { color: #0dcaf0; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #dee2e6; }
        th { background-color: #f8f9fa; font-weight: 600; }
        .metric-badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; margin: 2px; }
        .metric-passed { background-color: #d1edff; color: #0c63e4; }
        .metric-failed { background-color: #f8d7da; color: #721c24; }
        .case-passed { background-color: #d1f2eb; }
        .case-failed { background-color: #f8d7da; }
        .truncate { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ run.name }}</h1>
        <p><strong>Agent:</strong> {{ run.agent_version.agent.name }} ({{ run.agent_version.display_name }})</p>
        <p><strong>Dataset:</strong> {{ run.dataset_version.dataset.name }} ({{ run.dataset_version.display_name }})</p>
        <p><strong>Generated:</strong> {{ generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h3 class="{% if run.overall_score >= 0.8 %}success{% elif run.overall_score >= 0.6 %}warning{% else %}danger{% endif %}">
                {{ "%.2f"|format(run.overall_score) if run.overall_score else 'N/A' }}
            </h3>
            <div class="label">Overall Score</div>
        </div>
        <div class="summary-card">
            <h3 class="{% if run.pass_rate >= 0.8 %}success{% elif run.pass_rate >= 0.6 %}warning{% else %}danger{% endif %}">
                {{ "%.1f"|format(run.pass_rate * 100) if run.pass_rate else 'N/A' }}%
            </h3>
            <div class="label">Pass Rate</div>
        </div>
        <div class="summary-card">
            <h3 class="success">{{ passed_cases }}</h3>
            <div class="label">Passed Cases</div>
        </div>
        <div class="summary-card">
            <h3 class="danger">{{ failed_cases }}</h3>
            <div class="label">Failed Cases</div>
        </div>
    </div>

    {% if metric_summaries %}
    <h2>Metric Breakdown</h2>
    <div class="summary">
        {% for metric_name, summary in metric_summaries.items() %}
        <div class="summary-card">
            <h3 class="{% if (summary.passed_count / summary.total_count) >= 0.8 %}success{% elif (summary.passed_count / summary.total_count) >= 0.6 %}warning{% else %}danger{% endif %}">
                {{ "%.2f"|format(summary.scores|sum / summary.scores|length) }}
            </h3>
            <div class="label">{{ metric_name|replace('_', ' ')|title }}</div>
            <small>{{ summary.passed_count }}/{{ summary.total_count }} passed</small>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <h2>Test Case Results</h2>
    <table>
        <thead>
            <tr>
                <th>Case ID</th>
                <th>Question</th>
                <th>Answer</th>
                <th>Score</th>
                <th>Result</th>
                <th>Metrics</th>
            </tr>
        </thead>
        <tbody>
            {% for test_case in test_cases %}
            <tr class="{% if test_case.passed %}case-passed{% else %}case-failed{% endif %}">
                <td><code>{{ test_case.case_id }}</code></td>
                <td class="truncate">{{ test_case.question }}</td>
                <td class="truncate">{{ test_case.actual_answer }}</td>
                <td>
                    {% if test_case.overall_score is not none %}
                        {{ "%.2f"|format(test_case.overall_score) }}
                    {% else %}
                        N/A
                    {% endif %}
                </td>
                <td>
                    {% if test_case.passed %}
                        <span class="metric-badge metric-passed">✓ PASS</span>
                    {% else %}
                        <span class="metric-badge metric-failed">✗ FAIL</span>
                    {% endif %}
                </td>
                <td>
                    {% for metric in test_case.metric_results %}
                        <span class="metric-badge {% if metric.passed %}metric-passed{% else %}metric-failed{% endif %}">
                            {{ metric.metric_name.replace('_', ' ')|title }}: {{ "%.2f"|format(metric.score) }}
                        </span>
                    {% endfor %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <div class="footer">
        <p>Report generated by Simple Eval on {{ generated_at.strftime('%Y-%m-%d at %H:%M:%S') }}</p>
    </div>
</body>
</html>
        ''')
        
        html_content = html_template.render(**template_data)
        
        with open(filepath, 'w', encoding='utf-8') as htmlfile:
            htmlfile.write(html_content)
        
        return filepath

class ResultsFilter:
    """Handles filtering and searching test case results"""
    
    @staticmethod
    def filter_test_cases(test_cases: List[TestCase], filters: Dict[str, Any]) -> List[TestCase]:
        """Apply filters to test cases"""
        filtered = test_cases
        
        # Filter by status (passed/failed)
        if filters.get('status'):
            if filters['status'] == 'failed':
                filtered = [tc for tc in filtered if not tc.passed]
            elif filters['status'] == 'passed':
                filtered = [tc for tc in filtered if tc.passed]
        
        # Text search in question or answer
        if filters.get('search'):
            search_term = filters['search'].lower()
            filtered = [
                tc for tc in filtered 
                if (search_term in tc.question.lower() or 
                    search_term in tc.actual_answer.lower() or
                    search_term in tc.case_id.lower())
            ]
        
        # Filter by metric score range
        if filters.get('min_score') is not None:
            min_score = float(filters['min_score'])
            filtered = [tc for tc in filtered if tc.overall_score and tc.overall_score >= min_score]
        
        if filters.get('max_score') is not None:
            max_score = float(filters['max_score'])
            filtered = [tc for tc in filtered if tc.overall_score and tc.overall_score <= max_score]
        
        # Sort results
        sort_by = filters.get('sort_by', 'case_id')
        reverse = filters.get('sort_desc', False)
        
        if sort_by == 'score':
            filtered.sort(key=lambda tc: tc.overall_score or 0, reverse=reverse)
        elif sort_by == 'case_id':
            filtered.sort(key=lambda tc: tc.case_id, reverse=reverse)
        elif sort_by == 'status':
            filtered.sort(key=lambda tc: tc.passed or False, reverse=reverse)
        
        return filtered