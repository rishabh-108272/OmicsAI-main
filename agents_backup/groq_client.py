"""
Groq API Client for Multi-Agentic AI System
Provides fast LLM inference with retry logic and error handling
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import time

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Try to import groq, install if not available
try:
    from groq import Groq
except ImportError:
    logger.warning("Groq SDK not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "groq"])
    from groq import Groq


class GroqClient:
    """
    Singleton Groq client with retry logic and caching
    """
    
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize Groq client with API key"""
        # Try to load from environment
        load_dotenv()
        
        # Get Groq API key
        groq_api_key = os.environ.get('GROQ_API_KEY')
        
        if not groq_api_key:
            # Check for .env files in common locations
            env_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'multi agent rag', '.env'),
                os.path.join(os.path.dirname(__file__), '..', '..', '.env'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'),
            ]
            
            for env_path in env_paths:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    groq_api_key = os.environ.get('GROQ_API_KEY')
                    if groq_api_key:
                        break
        
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please set it in environment variables "
                "or create a .env file with GROQ_API_KEY=your_key"
            )
        
        self._client = Groq(api_key=groq_api_key)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_retries = 3
        self._retry_delay = 1.0  # seconds
        
        logger.info("GroqClient initialized successfully")
    
    def _get_cache_key(self, messages: List[Dict]) -> str:
        """Generate cache key from messages"""
        import hashlib
        content = str(messages)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, messages: List[Dict]) -> Optional[str]:
        """Get cached response"""
        cache_key = self._get_cache_key(messages)
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            # Cache expires after 1 hour
            if (datetime.now() - entry['timestamp']).seconds < 3600:
                logger.info(f"Cache hit for query")
                return entry['response']
            else:
                del self._cache[cache_key]
        return None
    
    def _add_to_cache(self, messages: List[Dict], response: str):
        """Add response to cache"""
        cache_key = self._get_cache_key(messages)
        self._cache[cache_key] = {
            'response': response,
            'timestamp': datetime.now()
        }
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        use_cache: bool = True,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send chat completion request with retry logic
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Groq model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            use_cache: Whether to use caching
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Dict with 'content', 'model', 'usage', etc.
        """
        # Add system prompt if provided
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            full_messages = messages
        
        # Check cache
        if use_cache:
            cached = self._get_from_cache(full_messages)
            if cached:
                return {
                    'content': cached,
                    'model': model,
                    'cached': True
                }
        
        # Retry logic
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=None
                )
                
                result = {
                    'content': response.choices[0].message.content,
                    'model': response.model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    },
                    'cached': False
                }
                
                # Add to cache
                if use_cache:
                    self._add_to_cache(full_messages, result['content'])
                
                logger.info(f"Groq API call successful: {result['usage']}")
                return result
                
            except Exception as e:
                logger.warning(f"Groq API attempt {attempt + 1} failed: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Groq API failed after {self._max_retries} attempts: {e}")
    
    def structured_completion(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        model: str = "llama-3.3-70b-versatile",
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get structured JSON output from LLM
        
        Args:
            messages: List of message dicts
            schema: JSON schema for expected output
            model: Groq model to use
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON response
        """
        import json
        
        # Create a prompt that requests JSON
        schema_str = json.dumps(schema, indent=2)
        
        full_messages = messages.copy()
        if system_prompt:
            full_messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Add instruction to output JSON
        instruction = f"""Please respond with valid JSON only, matching this schema:
{schema_str}

Respond with ONLY the JSON, no other text."""
        
        # Add as user message if no messages
        if not full_messages:
            full_messages = [{"role": "user", "content": instruction}]
        else:
            # Append to last user message
            for msg in reversed(full_messages):
                if msg['role'] == 'user':
                    msg['content'] += f"\n\n{instruction}"
                    break
        
        response = self.chat_completion(
            messages=full_messages,
            model=model,
            temperature=0.1,  # Low temperature for structured output
            max_tokens=4096
        )
        
        # Parse JSON
        try:
            return json.loads(response['content'])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response['content'], re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise
    
    def clear_cache(self):
        """Clear all cached responses"""
        self._cache.clear()
        logger.info("GroqClient cache cleared")


# Singleton instance
def get_groq_client() -> GroqClient:
    """Get singleton GroqClient instance"""
    return GroqClient()

