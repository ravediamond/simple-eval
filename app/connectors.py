import asyncio
import aiohttp
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ConnectorResponse:
    """Response from a connector evaluation"""
    answer: str
    success: bool
    error_message: Optional[str] = None
    response_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ConnectorConfig:
    """Configuration for a connector"""
    connector_type: str
    endpoint_url: str
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout_seconds: int = 30
    max_retries: int = 3
    rate_limit_per_minute: int = 60
    additional_params: Optional[Dict[str, Any]] = None

class ConnectorError(Exception):
    """Base exception for connector errors"""
    pass

class RateLimitExceededError(ConnectorError):
    """Raised when rate limit is exceeded"""
    pass

class TimeoutError(ConnectorError):
    """Raised when request times out"""
    pass

class BaseConnector(ABC):
    """Abstract base class for all connectors"""
    
    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.request_history: List[datetime] = []
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    def _check_rate_limit(self):
        """Check if we're within rate limits"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Remove old requests
        self.request_history = [req_time for req_time in self.request_history if req_time > minute_ago]
        
        if len(self.request_history) >= self.config.rate_limit_per_minute:
            raise RateLimitExceededError(f"Rate limit of {self.config.rate_limit_per_minute} requests per minute exceeded")
        
        self.request_history.append(now)
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except (aiohttp.ClientError, TimeoutError) as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    # Exponential backoff: 1s, 2s, 4s, etc.
                    delay = 2 ** attempt
                    await asyncio.sleep(delay)
                else:
                    break
        
        raise ConnectorError(f"Failed after {self.config.max_retries + 1} attempts: {last_exception}")
    
    @abstractmethod
    async def evaluate(self, question: str, context: Optional[str] = None) -> ConnectorResponse:
        """Evaluate a question and return the response"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the connector is working properly"""
        pass

class OpenAIConnector(BaseConnector):
    """Connector for OpenAI-compatible APIs"""
    
    async def evaluate(self, question: str, context: Optional[str] = None) -> ConnectorResponse:
        """Evaluate using OpenAI-compatible API"""
        self._check_rate_limit()
        
        start_time = time.time()
        
        try:
            return await self._retry_with_backoff(self._make_request, question, context)
        except Exception as e:
            return ConnectorResponse(
                answer="",
                success=False,
                error_message=str(e),
                response_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _make_request(self, question: str, context: Optional[str] = None) -> ConnectorResponse:
        """Make the actual API request"""
        start_time = time.time()
        
        # Prepare the prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        # Prepare request payload
        payload = {
            "model": self.config.model_name or "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        # Add any additional parameters
        if self.config.additional_params:
            payload.update(self.config.additional_params)
        
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        async with self._session.post(
            self.config.endpoint_url,
            json=payload,
            headers=headers
        ) as response:
            response_time_ms = int((time.time() - start_time) * 1000)
            
            if response.status != 200:
                error_text = await response.text()
                return ConnectorResponse(
                    answer="",
                    success=False,
                    error_message=f"HTTP {response.status}: {error_text}",
                    response_time_ms=response_time_ms
                )
            
            data = await response.json()
            
            # Extract answer from response
            try:
                answer = data["choices"][0]["message"]["content"].strip()
                tokens_used = data.get("usage", {}).get("total_tokens")
                
                return ConnectorResponse(
                    answer=answer,
                    success=True,
                    response_time_ms=response_time_ms,
                    tokens_used=tokens_used,
                    metadata={"raw_response": data}
                )
            except (KeyError, IndexError) as e:
                return ConnectorResponse(
                    answer="",
                    success=False,
                    error_message=f"Failed to parse response: {e}",
                    response_time_ms=response_time_ms,
                    metadata={"raw_response": data}
                )
    
    async def test_connection(self) -> bool:
        """Test the OpenAI connection with a simple request"""
        try:
            response = await self.evaluate("What is 2+2?")
            return response.success
        except Exception:
            return False

class HTTPConnector(BaseConnector):
    """Generic HTTP connector for custom APIs"""
    
    async def evaluate(self, question: str, context: Optional[str] = None) -> ConnectorResponse:
        """Evaluate using generic HTTP API"""
        self._check_rate_limit()
        
        start_time = time.time()
        
        try:
            return await self._retry_with_backoff(self._make_request, question, context)
        except Exception as e:
            return ConnectorResponse(
                answer="",
                success=False,
                error_message=str(e),
                response_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _make_request(self, question: str, context: Optional[str] = None) -> ConnectorResponse:
        """Make the actual API request"""
        start_time = time.time()
        
        # Prepare request payload - customizable format
        payload = {
            "question": question,
            "context": context,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        # Add any additional parameters
        if self.config.additional_params:
            payload.update(self.config.additional_params)
        
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        async with self._session.post(
            self.config.endpoint_url,
            json=payload,
            headers=headers
        ) as response:
            response_time_ms = int((time.time() - start_time) * 1000)
            
            if response.status != 200:
                error_text = await response.text()
                return ConnectorResponse(
                    answer="",
                    success=False,
                    error_message=f"HTTP {response.status}: {error_text}",
                    response_time_ms=response_time_ms
                )
            
            data = await response.json()
            
            # Extract answer from response - flexible field mapping
            try:
                # Try common field names for the answer
                answer = None
                for field in ["answer", "response", "text", "content", "result"]:
                    if field in data:
                        answer = str(data[field]).strip()
                        break
                
                if answer is None:
                    # If no standard field found, try to get the first string value
                    for value in data.values():
                        if isinstance(value, str):
                            answer = value.strip()
                            break
                
                if answer is None:
                    return ConnectorResponse(
                        answer="",
                        success=False,
                        error_message="No answer field found in response",
                        response_time_ms=response_time_ms,
                        metadata={"raw_response": data}
                    )
                
                return ConnectorResponse(
                    answer=answer,
                    success=True,
                    response_time_ms=response_time_ms,
                    metadata={"raw_response": data}
                )
            except Exception as e:
                return ConnectorResponse(
                    answer="",
                    success=False,
                    error_message=f"Failed to parse response: {e}",
                    response_time_ms=response_time_ms,
                    metadata={"raw_response": data}
                )
    
    async def test_connection(self) -> bool:
        """Test the HTTP connection with a simple request"""
        try:
            response = await self.evaluate("Test question")
            return response.success
        except Exception:
            return False

class ConnectorFactory:
    """Factory for creating connectors"""
    
    _connectors = {
        "openai": OpenAIConnector,
        "http": HTTPConnector,
    }
    
    @classmethod
    def create_connector(cls, config: ConnectorConfig) -> BaseConnector:
        """Create a connector instance based on config"""
        connector_class = cls._connectors.get(config.connector_type)
        if not connector_class:
            raise ValueError(f"Unknown connector type: {config.connector_type}")
        
        return connector_class(config)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available connector types"""
        return list(cls._connectors.keys())

class ConnectorManager:
    """Manages connector instances and concurrent evaluations"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_batch(
        self,
        connector: BaseConnector,
        questions: List[Dict[str, Any]]
    ) -> List[ConnectorResponse]:
        """Evaluate a batch of questions with concurrency control"""
        
        async def evaluate_single(question_data):
            async with self.semaphore:
                return await connector.evaluate(
                    question_data["question"],
                    question_data.get("context")
                )
        
        tasks = [evaluate_single(q) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ConnectorResponse(
                    answer="",
                    success=False,
                    error_message=f"Evaluation failed: {str(result)}"
                ))
            else:
                processed_results.append(result)
        
        return processed_results