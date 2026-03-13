"""
Agent Orchestrator for Multi-Agentic AI Validation System
Coordinates all validation agents and synthesizes results
"""

import json
import time
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_agent import (
    BaseAgent, 
    ValidationResult, 
    ValidationStatus, 
    ConfidenceLevel,
    ValidationCheck
)
from .pathway_reasoning_agent import get_pathway_reasoning_agent
from .drug_association_agent import get_drug_association_agent
from .literature_evidence_agent import get_literature_evidence_agent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrates multiple validation agents for comprehensive analysis
    
    Coordinates NEW agents:
    1. Pathway Reasoning Agent (KEGG/UniProt pathway enrichment)  
    2. Drug Association Agent (DrugBank/ClinicalTrials drug links)
    3. Literature Evidence Agent (LIVE PubMed evidence)
    
    Features:
    - Parallel agent execution
    - Result aggregation
    - Cross-agent consensus
    - Unified reporting
    """
    
    def __init__(self, enable_parallel: bool = True):
        """
        Initialize orchestrator
        
        Args:
            enable_parallel: Whether to run agents in parallel
        """
        self.enable_parallel = enable_parallel
        self.logger = logging.getLogger(__name__)
        
        # Initialize NEW agents
        self.agents = {
            'pathway_reasoning': get_pathway_reasoning_agent(),
            'drug_association': get_drug_association_agent(),
            'literature_evidence': get_literature_evidence_agent()
        }
        
        self.logger.info("AgentOrchestrator initialized with agents: %s", 
                        list(self.agents.keys()))
    
    def validate_all(
        self, 
        data: Dict[str, Any],
        agent_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run validation with all or selected agents
        
        Args:
            data: Data to validate
            agent_types: Optional list of agent types to run
            
        Returns:
            Dict with aggregated validation results
        """
        start_time = time.time()
        
        # Determine which agents to run
        if agent_types is None:
            agent_types = list(self.agents.keys())
        
        # Run agents
        if self.enable_parallel:
            results = self._run_agents_parallel(data, agent_types)
        else:
            results = self._run_agents_sequential(data, agent_types)
        
        # Aggregate results
        aggregated = self._aggregate_results(results)
        
        # Generate cross-agent insights
        cross_agent = self._generate_cross_agent_insights(results, data)
        aggregated['cross_agent_insights'] = cross_agent
        
        # Add metadata
        aggregated['orchestrator_metadata'] = {
            'total_processing_time': time.time() - start_time,
            'agents_run': list(results.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        return aggregated
    
    def _run_agents_parallel(
        self, 
        data: Dict[str, Any], 
        agent_types: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Run agents in parallel"""
        results = {}
        
        def run_agent(agent_type: str):
            try:
                agent = self.agents.get(agent_type)
                if agent is None:
                    return agent_type, {
                        'error': f'Unknown agent type: {agent_type}',
                        'status': 'error'
                    }
                
                self.logger.info(f"Running agent: {agent_type}")
                result = agent.validate(data)
                return agent_type, result.to_dict()
                
            except Exception as e:
                self.logger.error(f"Agent {agent_type} failed: {e}")
                return agent_type, {
                    'error': str(e),
                    'status': 'error'
                }
        
        # Run agents in parallel
        with ThreadPoolExecutor(max_workers=len(agent_types)) as executor:
            futures = {
                executor.submit(run_agent, agent_type): agent_type 
                for agent_type in agent_types
            }
            
            for future in as_completed(futures):
                agent_type = futures[future]
                try:
                    key, result = future.result()
                    results[key] = result
                except Exception as e:
                    self.logger.error(f"Future failed for {agent_type}: {e}")
                    results[agent_type] = {
                        'error': str(e),
                        'status': 'error'
                    }
        
        return results
    
    def _run_agents_sequential(
        self, 
        data: Dict[str, Any], 
        agent_types: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Run agents sequentially"""
        results = {}
        
        for agent_type in agent_types:
            try:
                agent = self.agents.get(agent_type)
                if agent is None:
                    results[agent_type] = {
                        'error': f'Unknown agent type: {agent_type}',
                        'status': 'error'
                    }
                    continue
                
                self.logger.info(f"Running agent: {agent_type}")
                result = agent.validate(data)
                results[agent_type] = result.to_dict()
                
            except Exception as e:
                self.logger.error(f"Agent {agent_type} failed: {e}")
                results[agent_type] = {
                    'error': str(e),
                    'status': 'error'
                }
        
        return results
    
    def _aggregate_results(
        self, 
        agent_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results from all agents"""
        
        # Count statuses
        status_counts = {
            'passed': 0,
            'failed': 0,
            'warning': 0,
            'error': 0,
            'skipped': 0
        }
        
        confidence_levels = []
        all_checks = []
        
        for agent_type, result in agent_results.items():
            status = result.get('overall_status', 'error')
            if status in status_counts:
                status_counts[status] += 1
            
            confidence = result.get('overall_confidence', 'none')
            if confidence != 'none':
                confidence_levels.append(confidence)
            
            # Collect checks
            checks = result.get('checks', [])
            for check in checks:
                check['agent'] = agent_type
                all_checks.append(check)
        
        # Determine overall status
        if status_counts['error'] > 0:
            overall_status = ValidationStatus.ERROR
        elif status_counts['failed'] > 0:
            overall_status = ValidationStatus.FAILED
        elif status_counts['warning'] > 0:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED
        
        # Determine overall confidence
        if not confidence_levels:
            overall_confidence = ConfidenceLevel.NONE
        else:
            high_count = sum(1 for c in confidence_levels if c == 'high')
            if high_count / len(confidence_levels) >= 0.7:
                overall_confidence = ConfidenceLevel.HIGH
            elif high_count / len(confidence_levels) >= 0.3:
                overall_confidence = ConfidenceLevel.MEDIUM
            else:
                overall_confidence = ConfidenceLevel.LOW
        
        # Generate summary
        summary = self._generate_aggregated_summary(status_counts)
        
        # Collect all recommendations
        recommendations = []
        for result in agent_results.values():
            recs = result.get('recommendations', [])
            recommendations.extend(recs)
        
        # Deduplicate recommendations
        recommendations = list(dict.fromkeys(recommendations))
        
        return {
            'overall_status': overall_status.value,
            'overall_confidence': overall_confidence.value,
            'summary': summary,
            'status_counts': status_counts,
            'agent_results': agent_results,
            'all_checks': all_checks,
            'recommendations': recommendations
        }
    
    def _generate_aggregated_summary(self, status_counts: Dict[str, int]) -> str:
        """Generate summary from status counts"""
        total = sum(status_counts.values())
        
        if total == 0:
            return "No validation results available"
        
        passed = status_counts.get('passed', 0)
        warnings = status_counts.get('warning', 0)
        failed = status_counts.get('failed', 0)
        errors = status_counts.get('error', 0)
        
        parts = []
        if passed > 0:
            parts.append(f"{passed} passed")
        if warnings > 0:
            parts.append(f"{warnings} warnings")
        if failed > 0:
            parts.append(f"{failed} failed")
        if errors > 0:
            parts.append(f"{errors} errors")
        
        return "Validation: " + ", ".join(parts)
    
    def _generate_cross_agent_insights(
        self, 
        agent_results: Dict[str, Dict[str, Any]],
        original_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate insights across agents"""
        
        insights = {
            'consensus_findings': [],
            'conflicting_findings': [],
            'high_confidence_findings': [],
            'actionable_recommendations': []
        }
        
        # Collect all high confidence checks
        high_confidence_checks = []
        
        for agent_type, result in agent_results.items():
            checks = result.get('checks', [])
            for check in checks:
                if check.get('confidence') == 'high':
                    high_confidence_checks.append({
                        'agent': agent_type,
                        'check': check
                    })
        
        # Extract insights from high confidence checks
        for hc in high_confidence_checks:
            check = hc['check']
            if check.get('status') == 'passed':
                insights['high_confidence_findings'].append({
                    'agent': hc['agent'],
                    'finding': check.get('message', ''),
                    'evidence': check.get('evidence', {})
                })
        
        # Generate actionable recommendations based on consensus
        all_checks_by_name = {}
        for agent_type, result in agent_results.items():
            for check in result.get('checks', []):
                name = check.get('name', '')
                if name not in all_checks_by_name:
                    all_checks_by_name[name] = []
                all_checks_by_name[name].append({
                    'agent': agent_type,
                    'status': check.get('status'),
                    'message': check.get('message', '')
                })
        
        # Find consensus (same status from multiple agents)
        for check_name, checks in all_checks_by_name.items():
            if len(checks) >= 2:
                statuses = set(c['status'] for c in checks)
                if len(statuses) == 1:  # All agree
                    status = list(statuses)[0]
                    if status == 'failed':
                        insights['conflicting_findings'].append({
                            'check': check_name,
                            'message': f"All agents agree: {checks[0]['message']}",
                            'agents': [c['agent'] for c in checks]
                        })
                    elif status == 'passed':
                        insights['consensus_findings'].append({
                            'check': check_name,
                            'message': f"All agents agree: {checks[0]['message']}",
                            'agents': [c['agent'] for c in checks]
                        })
        
        # Generate actionable recommendations
        for agent_type, result in agent_results.items():
            recommendations = result.get('recommendations', [])
            for rec in recommendations:
                insights['actionable_recommendations'].append({
                    'agent': agent_type,
                    'recommendation': rec
                })
        
        return insights
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent types - NEW: pathway_reasoning, drug_association, literature_evidence"""
        return list(self.agents.keys())

    # Use validate_all(data, agent_types=['pathway_reasoning']) for single agents


# Singleton instance
_orchestrator = None

def get_orchestrator() -> AgentOrchestrator:
    """Get singleton AgentOrchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


# Django view integration
def run_validation_orchestrator(request):
    """
    Django view for running validation orchestrator
    
    Expected POST data:
    {
        'validation_type': 'classification' | 'biomarker' | 'drug' | 'protein' | 'all',
        'data': {...}  // Data to validate
    }
    """
    import json
    from django.http import JsonResponse
    from django.views.decorators.csrf import csrf_exempt
    from django.views.decorators.http import require_http_methods
    
    @csrf_exempt
    @require_http_methods(["POST"])
    def inner(request):
        try:
            body = json.loads(request.body.decode('utf-8'))
        except Exception:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        
        validation_type = body.get('validation_type', 'all')
        data = body.get('data', {})
        
        if not data:
            return JsonResponse({'error': 'No data provided'}, status=400)
        
        orchestrator = get_orchestrator()
        
        # Determine which agents to run
        if validation_type == 'all':
            agent_types = None
        elif validation_type in orchestrator.get_available_agents():
            agent_types = [validation_type]
        else:
            return JsonResponse({
                'error': f'Invalid validation type: {validation_type}',
                'available_types': orchestrator.get_available_agents()
            }, status=400)
        
        try:
            result = orchestrator.validate_all(data, agent_types)
            return JsonResponse(result, status=200)
        except Exception as e:
            logger.error(f"Validation orchestrator error: {e}")
            return JsonResponse({
                'error': f'Validation failed: {str(e)}'
            }, status=500)
    
    return inner(request)

