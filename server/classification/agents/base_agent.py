"""
Base Agent Framework for Multi-Agentic AI Validation System
Provides abstract base class and common validation result structures
"""

import logging
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class ConfidenceLevel(Enum):
    """Confidence levels for validation"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class ValidationCheck:
    """Individual validation check result"""
    name: str
    status: ValidationStatus
    message: str
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    evidence: Optional[Dict[str, Any]] = None
    details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'confidence': self.confidence.value,
            'evidence': self.evidence,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ValidationResult:
    """
    Complete validation result from an agent
    """
    agent_name: str
    overall_status: ValidationStatus
    overall_confidence: ConfidenceLevel
    summary: str
    checks: List[ValidationCheck] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'agent_name': self.agent_name,
            'overall_status': self.overall_status.value,
            'overall_confidence': self.overall_confidence.value,
            'summary': self.summary,
            'checks': [check.to_dict() for check in self.checks],
            'recommendations': self.recommendations,
            'metadata': self.metadata,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat(),
            'error': self.error
        }
    
    def add_check(self, check: ValidationCheck):
        """Add a validation check"""
        self.checks.append(check)
    
    def add_recommendation(self, recommendation: str):
        """Add a recommendation"""
        self.recommendations.append(recommendation)
    
    @property
    def passed_checks_count(self) -> int:
        """Count of passed checks"""
        return sum(1 for c in self.checks if c.status == ValidationStatus.PASSED)
    
    @property
    def failed_checks_count(self) -> int:
        """Count of failed checks"""
        return sum(1 for c in self.checks if c.status == ValidationStatus.FAILED)
    
    @property
    def warning_checks_count(self) -> int:
        """Count of warning checks"""
        return sum(1 for c in self.checks if c.status == ValidationStatus.WARNING)


class BaseAgent(ABC):
    """
    Abstract base class for all validation agents
    
    Each agent should:
    1. Implement the validate() method
    2. Define its own system prompt for the LLM
    3. Provide specialized validation logic
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the agent
        
        Args:
            name: Agent name
            description: Agent description
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._groq_client = None
        
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        System prompt that defines the agent's role and expertise
        Must be implemented by subclasses
        """
        pass
    
    @property
    def groq_client(self):
        """Get Groq client (lazy initialization)"""
        if self._groq_client is None:
            from .groq_client import get_groq_client
            self._groq_client = get_groq_client()
        return self._groq_client
    
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate the given data
        
        Args:
            data: Data to validate (specific to each agent)
            
        Returns:
            ValidationResult with detailed validation results
        """
        pass
    
    def _create_result(
        self,
        status: ValidationStatus,
        summary: str,
        checks: Optional[List[ValidationCheck]] = None,
        recommendations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        processing_time: float = 0.0,
        error: Optional[str] = None
    ) -> ValidationResult:
        """Helper to create ValidationResult"""
        # Determine overall confidence
        if status == ValidationStatus.PASSED:
            confidence = ConfidenceLevel.HIGH
        elif status == ValidationStatus.WARNING:
            confidence = ConfidenceLevel.MEDIUM
        elif status == ValidationStatus.FAILED:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.NONE
            
        return ValidationResult(
            agent_name=self.name,
            overall_status=status,
            overall_confidence=confidence,
            summary=summary,
            checks=checks or [],
            recommendations=recommendations or [],
            metadata=metadata or {},
            processing_time=processing_time,
            error=error
        )
    
    def _create_check(
        self,
        name: str,
        status: ValidationStatus,
        message: str,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        evidence: Optional[Dict[str, Any]] = None,
        details: Optional[str] = None
    ) -> ValidationCheck:
        """Helper to create ValidationCheck"""
        return ValidationCheck(
            name=name,
            status=status,
            message=message,
            confidence=confidence,
            evidence=evidence,
            details=details
        )
    
    def _query_llm(
        self,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        """
        Query the LLM with the agent's system prompt
        
        Args:
            user_message: User query
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response content
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self.groq_client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response['content']
    
    def _query_llm_structured(
        self,
        user_message: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Query LLM with structured output
        
        Args:
            user_message: User query
            schema: Expected JSON schema
            
        Returns:
            Parsed JSON response
        """
        response = self.groq_client.structured_completion(
            messages=[{"role": "user", "content": user_message}],
            schema=schema,
            system_prompt=self.system_prompt
        )
        
        return response
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)


class ReasoningAgent(BaseAgent):
    """
    Extended agent that uses structured reasoning for validation
    Implements chain-of-thought reasoning
    """
    
    @property
    def reasoning_template(self) -> str:
        """
        Template for chain-of-thought reasoning
        Should be implemented by subclasses
        """
        return """Think step by step about the validation:
1. First, analyze the input data
2. Then, identify key validation criteria
3. Next, check each criterion
4. Finally, summarize findings"""
    
    def validate_with_reasoning(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate with chain-of-thought reasoning
        
        This method:
        1. Asks LLM to reason about the validation
        2. Extracts structured validation checks from reasoning
        3. Returns comprehensive validation result
        """
        import time
        start_time = time.time()
        
        try:
            # Prepare reasoning prompt
            data_json = json.dumps(data, indent=2, default=str)
            reasoning_prompt = f"""
{self.reasoning_template}

Input data to validate:
{data_json}

Please analyze this data and provide:
1. Step-by-step reasoning
2. List of validation checks to perform
3. Expected outcomes
4. Final assessment
"""
            
            # Get LLM reasoning
            reasoning = self._query_llm(reasoning_prompt, temperature=0.5, max_tokens=2048)
            
            # Parse reasoning into structured checks
            checks = self._parse_reasoning_to_checks(reasoning)
            
            # Determine overall status from checks
            if any(c.status == ValidationStatus.FAILED for c in checks):
                overall_status = ValidationStatus.FAILED
            elif any(c.status == ValidationStatus.WARNING for c in checks):
                overall_status = ValidationStatus.WARNING
            else:
                overall_status = ValidationStatus.PASSED
            
            processing_time = time.time() - start_time
            
            return self._create_result(
                status=overall_status,
                summary=f"Validation completed with {len(checks)} checks",
                checks=checks,
                processing_time=processing_time,
                metadata={'reasoning': reasoning}
            )
            
        except Exception as e:
            self.log_error(f"Validation failed: {e}")
            processing_time = time.time() - start_time
            
            return self._create_result(
                status=ValidationStatus.ERROR,
                summary=f"Validation error: {str(e)}",
                processing_time=processing_time,
                error=str(e)
            )
    
    def _parse_reasoning_to_checks(self, reasoning: str) -> List[ValidationCheck]:
        """
        Parse LLM reasoning into structured validation checks
        
        This is a simplified implementation. Subclasses can override
        for more specific parsing.
        """
        # Default: create a single check based on reasoning
        check = self._create_check(
            name="LLM Reasoning Analysis",
            status=ValidationStatus.PASSED,
            message="Reasoning-based validation completed",
            confidence=ConfidenceLevel.MEDIUM,
            evidence={"reasoning_summary": reasoning[:500]}
        )
        
        return [check]

