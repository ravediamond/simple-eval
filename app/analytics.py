import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from google.cloud import firestore
from google.cloud.firestore import Client
import logging

logger = logging.getLogger(__name__)

class AnalyticsTracker:
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.db: Optional[Client] = None
        self._init_firestore()
    
    def _init_firestore(self):
        """Initialize Firestore client"""
        try:
            if self.project_id:
                # Use named database for new deployments
                database_name = os.getenv("FIRESTORE_DATABASE", "evalnow-db")
                self.db = firestore.Client(project=self.project_id, database=database_name)
                logger.info(f"Firestore client initialized successfully (database: {database_name})")
            else:
                logger.warning("GOOGLE_CLOUD_PROJECT not set, analytics will be disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore: {e}")
            self.db = None
    
    def _is_enabled(self) -> bool:
        """Check if analytics is enabled"""
        return self.db is not None
    
    async def track_event(self, event_type: str, user_id: str, data: Dict[str, Any] = None):
        """Track an analytics event"""
        if not self._is_enabled():
            return
        
        try:
            event_data = {
                'event_type': event_type,
                'user_id': user_id,
                'timestamp': datetime.now(timezone.utc),
                'data': data or {}
            }
            
            # Add to events collection
            self.db.collection('analytics_events').add(event_data)
            
            # Update user metrics
            await self._update_user_metrics(user_id, event_type, data)
            
        except Exception as e:
            logger.error(f"Failed to track event {event_type}: {e}")
    
    async def _update_user_metrics(self, user_id: str, event_type: str, data: Dict[str, Any]):
        """Update user-specific metrics"""
        try:
            user_ref = self.db.collection('user_metrics').document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
            else:
                user_data = {
                    'first_seen': datetime.now(timezone.utc),
                    'total_visits': 0,
                    'total_evaluations': 0,
                    'total_pdf_downloads': 0,
                    'dataset_sizes': [],
                    'last_activity': datetime.now(timezone.utc)
                }
            
            # Update based on event type
            if event_type == 'page_visit':
                user_data['total_visits'] += 1
            elif event_type == 'dataset_uploaded':
                user_data['total_evaluations'] += 1
                if data and 'dataset_size' in data:
                    user_data['dataset_sizes'].append(data['dataset_size'])
                # Track token usage
                if 'total_tokens' in data:
                    if 'total_tokens_used' not in user_data:
                        user_data['total_tokens_used'] = 0
                        user_data['total_input_tokens'] = 0
                        user_data['total_output_tokens'] = 0
                    user_data['total_tokens_used'] += data.get('total_tokens', 0)
                    user_data['total_input_tokens'] += data.get('input_tokens', 0)
                    user_data['total_output_tokens'] += data.get('output_tokens', 0)
            elif event_type == 'pdf_downloaded':
                user_data['total_pdf_downloads'] += 1
            
            user_data['last_activity'] = datetime.now(timezone.utc)
            
            # Upsert user metrics
            user_ref.set(user_data)
            
        except Exception as e:
            logger.error(f"Failed to update user metrics for {user_id}: {e}")
    
    def calculate_llm_cost(self, input_tokens: int, output_tokens: int, judge_model: str, analysis_model: str) -> float:
        """Calculate cost based on token usage and model pricing"""
        # Gemini 2.5 Flash pricing (per 1M tokens)
        # Input: $0.075, Output: $0.30
        GEMINI_25_FLASH_INPUT_COST = 0.075 / 1_000_000
        GEMINI_25_FLASH_OUTPUT_COST = 0.30 / 1_000_000
        
        # Gemini 2.5 Flash Lite pricing (per 1M tokens)  
        # Input: $0.0375, Output: $0.15 (assumed half price of regular flash)
        GEMINI_25_FLASH_LITE_INPUT_COST = 0.0375 / 1_000_000
        GEMINI_25_FLASH_LITE_OUTPUT_COST = 0.15 / 1_000_000
        
        # Estimate: 80% of tokens are from judge model (many evaluations), 20% from analysis
        judge_input_tokens = int(input_tokens * 0.8)
        judge_output_tokens = int(output_tokens * 0.8)
        analysis_input_tokens = int(input_tokens * 0.2)
        analysis_output_tokens = int(output_tokens * 0.2)
        
        judge_cost = (judge_input_tokens * GEMINI_25_FLASH_LITE_INPUT_COST + 
                     judge_output_tokens * GEMINI_25_FLASH_LITE_OUTPUT_COST)
        analysis_cost = (analysis_input_tokens * GEMINI_25_FLASH_INPUT_COST + 
                        analysis_output_tokens * GEMINI_25_FLASH_OUTPUT_COST)
        
        return judge_cost + analysis_cost

    async def get_kpi_summary(self) -> Dict[str, Any]:
        """Get KPI summary for dashboard"""
        if not self._is_enabled():
            return {"error": "Analytics not enabled"}
        
        try:
            # Get current month boundaries
            now = datetime.now(timezone.utc)
            current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Get all user metrics
            users_ref = self.db.collection('user_metrics')
            users = list(users_ref.stream())
            
            # Get current month events
            events_ref = self.db.collection('analytics_events')
            current_month_events = list(events_ref.where('timestamp', '>=', current_month_start).stream())
            
            total_users = len(users)
            active_evaluators = 0
            repeat_users = 0
            total_evaluations = 0
            total_pdf_downloads = 0
            dataset_sizes = []
            total_tokens_used = 0
            total_input_tokens = 0
            total_output_tokens = 0
            
            # Current month metrics
            current_month_tokens = 0
            current_month_input_tokens = 0
            current_month_output_tokens = 0
            
            for user_doc in users:
                user_data = user_doc.to_dict()
                evals = user_data.get('total_evaluations', 0)
                
                if evals > 0:
                    active_evaluators += 1
                    total_evaluations += evals
                    
                if evals > 1:
                    repeat_users += 1
                
                total_pdf_downloads += user_data.get('total_pdf_downloads', 0)
                dataset_sizes.extend(user_data.get('dataset_sizes', []))
                
                # Aggregate token usage
                total_tokens_used += user_data.get('total_tokens_used', 0)
                total_input_tokens += user_data.get('total_input_tokens', 0)
                total_output_tokens += user_data.get('total_output_tokens', 0)
            
            # Process current month events for token usage
            for event_doc in current_month_events:
                event_data = event_doc.to_dict()
                if event_data.get('event_type') == 'dataset_uploaded':
                    event_tokens = event_data.get('data', {})
                    current_month_tokens += event_tokens.get('total_tokens', 0)
                    current_month_input_tokens += event_tokens.get('input_tokens', 0)
                    current_month_output_tokens += event_tokens.get('output_tokens', 0)
            
            # Calculate dataset size distribution
            size_distribution = {
                'toy': len([s for s in dataset_sizes if s < 10]),
                'small': len([s for s in dataset_sizes if 10 <= s < 100]),
                'medium': len([s for s in dataset_sizes if 100 <= s < 1000]),
                'large': len([s for s in dataset_sizes if s >= 1000])
            }
            
            # Calculate LLM costs
            total_cost = self.calculate_llm_cost(total_input_tokens, total_output_tokens, "gemini-2.5-flash-lite", "gemini-2.5-flash")
            current_month_cost = self.calculate_llm_cost(current_month_input_tokens, current_month_output_tokens, "gemini-2.5-flash-lite", "gemini-2.5-flash")
            
            return {
                'total_users': total_users,
                'active_evaluators': active_evaluators,
                'activation_rate': round(active_evaluators / total_users * 100, 1) if total_users > 0 else 0,
                'repeat_users': repeat_users,
                'retention_rate': round(repeat_users / active_evaluators * 100, 1) if active_evaluators > 0 else 0,
                'total_evaluations': total_evaluations,
                'total_pdf_downloads': total_pdf_downloads,
                'pdf_conversion_rate': round(total_pdf_downloads / total_evaluations * 100, 1) if total_evaluations > 0 else 0,
                'dataset_size_distribution': size_distribution,
                'avg_dataset_size': round(sum(dataset_sizes) / len(dataset_sizes), 1) if dataset_sizes else 0,
                'total_tokens_used': total_tokens_used,
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'avg_tokens_per_evaluation': round(total_tokens_used / total_evaluations, 1) if total_evaluations > 0 else 0,
                'current_month_tokens': current_month_tokens,
                'current_month_input_tokens': current_month_input_tokens,
                'current_month_output_tokens': current_month_output_tokens,
                'total_llm_cost': round(total_cost, 4),
                'current_month_llm_cost': round(current_month_cost, 4)
            }
            
        except Exception as e:
            logger.error(f"Failed to get KPI summary: {e}")
            return {"error": str(e)}