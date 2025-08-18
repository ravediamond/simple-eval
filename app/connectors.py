import asyncio
import aiohttp
import json
import time
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    from litellm import acompletion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

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

class LiteLLMConnector(BaseConnector):
    """Connector for LiteLLM-supported providers (OpenAI, Anthropic, Google, AWS, Ollama, etc.)"""
    
    def __init__(self, config: ConnectorConfig):
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM is not installed. Run: pip install litellm")
        super().__init__(config)
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup environment variables for LiteLLM authentication"""
        if self.config.api_key:
            # Map connector types to appropriate environment variables
            env_mapping = {
                "litellm_openai": "OPENAI_API_KEY",
                "litellm_anthropic": "ANTHROPIC_API_KEY", 
                "litellm_google": "GOOGLE_API_KEY",
                "litellm_aws": "AWS_ACCESS_KEY_ID",
                "litellm_azure": "AZURE_API_KEY",
                "litellm_ollama": "OLLAMA_API_KEY",
                "litellm_huggingface": "HUGGINGFACE_API_KEY"
            }
            
            env_var = env_mapping.get(self.config.connector_type)
            if env_var and not os.getenv(env_var):
                os.environ[env_var] = self.config.api_key
    
    async def evaluate(self, question: str, context: Optional[str] = None) -> ConnectorResponse:
        """Evaluate using LiteLLM-supported provider"""
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
        """Make the actual LiteLLM API request"""
        start_time = time.time()
        
        # Prepare the prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        # Prepare request parameters for LiteLLM
        params = {
            "model": self.config.model_name or self._get_default_model(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        # Add any additional parameters
        if self.config.additional_params:
            params.update(self.config.additional_params)
        
        # Handle custom endpoint for local models (like Ollama)
        if self.config.endpoint_url and self.config.connector_type == "litellm_ollama":
            params["api_base"] = self.config.endpoint_url
        
        try:
            # Debug logging for Google authentication
            if self.config.connector_type == "litellm_google":
                print(f"Debug - Model: {params['model']}")
                print(f"Debug - GOOGLE_API_KEY set: {'GOOGLE_API_KEY' in os.environ}")
                print(f"Debug - GOOGLE_APPLICATION_CREDENTIALS set: {'GOOGLE_APPLICATION_CREDENTIALS' in os.environ}")
            
            # Make async call to LiteLLM
            response = await acompletion(**params)
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract answer from response
            answer = response.choices[0].message.content.strip()
            tokens_used = getattr(response.usage, 'total_tokens', None) if hasattr(response, 'usage') else None
            
            return ConnectorResponse(
                answer=answer,
                success=True,
                response_time_ms=response_time_ms,
                tokens_used=tokens_used,
                metadata={
                    "model": response.model if hasattr(response, 'model') else params["model"],
                    "provider": self._extract_provider_from_model(params["model"])
                }
            )
            
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            return ConnectorResponse(
                answer="",
                success=False,
                error_message=f"LiteLLM request failed: {str(e)}",
                response_time_ms=response_time_ms
            )
    
    def _get_default_model(self) -> str:
        """Get default model based on connector type"""
        defaults = {
            "litellm_openai": "gpt-3.5-turbo",
            "litellm_anthropic": "claude-3-haiku-20240307",
            "litellm_google": "gemini/gemini-pro", 
            "litellm_aws": "anthropic.claude-3-haiku-20240307-v1:0",
            "litellm_azure": "azure/gpt-35-turbo",
            "litellm_ollama": "ollama/llama2",
            "litellm_huggingface": "huggingface/microsoft/DialoGPT-medium"
        }
        return defaults.get(self.config.connector_type, "gpt-3.5-turbo")
    
    def _extract_provider_from_model(self, model: str) -> str:
        """Extract provider name from model string"""
        if "/" in model:
            return model.split("/")[0]
        return self.config.connector_type.replace("litellm_", "")
    
    async def test_connection(self) -> bool:
        """Test the LiteLLM connection with a simple request"""
        try:
            response = await self.evaluate("What is 2+2?")
            return response.success
        except Exception:
            return False

class ConnectorFactory:
    """Factory for creating connectors"""
    
    _connectors = {
        "openai": OpenAIConnector,
        "http": HTTPConnector,
        "litellm_openai": LiteLLMConnector,
        "litellm_anthropic": LiteLLMConnector,
        "litellm_google": LiteLLMConnector,
        "litellm_aws": LiteLLMConnector,
        "litellm_azure": LiteLLMConnector,
        "litellm_ollama": LiteLLMConnector,
        "litellm_huggingface": LiteLLMConnector,
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