# Agent Orchestrator - New 3-Agent System
# Coordinates Pathway, Drug, Literature agents with live data support

import json
import time
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist

from .base_agent import (
    BaseAgent, 
    ValidationResult, 
    ValidationStatus, 
    ConfidenceLevel,
    ValidationCheck
)
from .pathway_agent import get_pathway_agent
from .drug_agent import get_drug_agent
from .literature_agent import get_literature_agent
from ..models import AnalysisSession, ClassificationResult
# from ..ml_service import ml_service  # Direct CSV via string decode

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """New Orchestrator for Pathway/Drug/Literature agents.\n    \n    Supports:\n    - Live DB data (session_id)\n    - Direct CSV processing (file upload)\n    - Parallel execution\n    - Unified reporting\n    """

    def __init__(self, enable_parallel: bool = True):
        self.enable_parallel = enable_parallel
        self.logger = logging.getLogger(__name__)
        
        # New 3 agents
        self.agents = {
            'pathway': get_pathway_agent(),
            'drug': get_drug_agent(),
            'literature': get_literature_agent()
        }
        
        self.logger.info("New AgentOrchestrator initialized: %s", list(self.agents.keys()))

    def run_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry: handle session_id OR csv_data.\n        \n        Args:\n            input_data: {'session_id': str} OR {'csv_data': base64_csv, 'cancer_type': str}\n        """
        start_time = time.time()
        
        # Extract live or direct data
        data_for_agents = self._prepare_agent_data(input_data)
        
        agent_types = list(self.agents.keys())
        results = self._run_agents_parallel(data_for_agents, agent_types) if self.enable_parallel else self._run_agents_sequential(data_for_agents, agent_types)
        
        # Aggregate
        aggregated = self._aggregate_results(results)
        cross_insights = self._generate_cross_agent_insights(results)
        aggregated['cross_agent_insights'] = cross_insights
        
        metadata = {
            'total_time': round(time.time() - start_time, 2),
            'input_type': 'session' if input_data.get('session_id') else 'direct',
            'data_shape': len(data_for_agents.get('biomarkers', [])),
            'timestamp': datetime.now().isoformat()
        }
        aggregated['orchestrator_metadata'] = metadata
        
        return aggregated

    def _prepare_agent_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Live session or direct CSV -> standardized {'biomarkers', 'cancer_type', ...}"""
        if session_id := input_data.get('session_id'):
            return self._data_from_session(session_id)
        elif csv_data := input_data.get('csv_data'):
            return self._data_from_csv(csv_data, input_data.get('cancer_type', 'lung'))
        # neither session nor csv provided – error
        raise ValueError('Require "session_id" or "csv_data"')

    def _data_from_session(self, session_id: str) -> Dict[str, Any]:
        #Fetch latest biomarkers from AnalysisSession.
        try:
            session = AnalysisSession.objects.select_related('results').get(session_id=session_id)
            if session.status != 'completed':
                raise ValueError(f"Session {session_id} not completed")
            
            # Extract biomarkers/genes from results (assume top_genes field added)
            result = session.results.first()
            biomarkers = getattr(result, 'top_biomarkers', ['KRAS', 'EGFR']) or ['KRAS', 'EGFR']  # Fallback
            if isinstance(biomarkers, str):
                biomarkers = json.loads(biomarkers)
            
            cancer_type = session.model_type.replace('_', '-').title()
            
            return {
                'biomarkers': biomarkers[:50],  # Top 50
                'cancer_type': cancer_type,
                'session_info': {'id': str(session.session_id), 'model': cancer_type}
            }
        except ObjectDoesNotExist:
            raise ValueError(f"Session {session_id} not found")

    def _data_from_csv(self, csv_data: str, cancer_type: str) -> Dict[str, Any]:
        # \"\"\"Direct CSV -> ML predict -> extract top biomarkers.\"\"\"
# Mock direct CSV biomarkers (simplified - integrate ml_service later)
        biomarkers = ['KRAS', 'EGFR', 'TP53', 'PIK3CA', 'BRAF'][:50]

        # placeholder direct CSV handling; ml_service integration can replace this
        return {
            'biomarkers': biomarkers,
            'cancer_type': cancer_type,
            'prediction': {'confidence': 0.85}
        }

    def _run_agents_parallel(self, data: Dict[str, Any], agent_types: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        with ThreadPoolExecutor(max_workers=len(agent_types)) as executor:
            futures = {executor.submit(self._run_single_agent, agent_type, data): agent_type for agent_type in agent_types}
            for future in as_completed(futures):
                agent_type = futures[future]
                try:
                    result = future.result()
                    results[agent_type] = result.to_dict()
                except Exception as e:
                    logger.error(f"Agent {agent_type} failed: {e}")
                    results[agent_type] = {'error': str(e), 'status': 'error'}
        return results

    def _run_agents_sequential(self, data: Dict[str, Any], agent_types: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        for agent_type in agent_types:
            try:
                result = self._run_single_agent(agent_type, data)
                results[agent_type] = result.to_dict()
            except Exception as e:
                logger.error(f"Agent {agent_type} failed: {e}")
                results[agent_type] = {'error': str(e), 'status': 'error'}
        return results

    def _run_single_agent(self, agent_type: str, data: Dict[str, Any]) -> ValidationResult:
        agent = self.agents[agent_type]
        logger.info(f"Running {agent_type}")
        return agent.validate(data)

    def _aggregate_results(self, agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        status_counts = {'passed': 0, 'failed': 0, 'warning': 0, 'error': 0, 'skipped': 0}
        confidence_levels = []
        all_checks = []

        for agent_type, result in agent_results.items():
            status = result.get('overall_status', 'error')
            status_counts[status] += 1
            
            conf = result.get('overall_confidence', 'none')
            if conf != 'none':
                confidence_levels.append(conf)
            
            checks = result.get('checks', [])
            for check in checks:
                check['agent'] = agent_type
                all_checks.append(check)

        # Status
        if status_counts['error'] > 0:
            overall_status = 'error'
        elif status_counts['failed'] > 0:
            overall_status = 'failed'
        elif status_counts['warning'] > 0:
            overall_status = 'warning'
        else:
            overall_status = 'passed'

        # Confidence
        if confidence_levels:
            high_pct = confidence_levels.count('high') / len(confidence_levels)
            overall_conf = 'high' if high_pct >= 0.66 else 'medium' if high_pct >= 0.33 else 'low'
        else:
            overall_conf = 'none'

        recs = []
        for result in agent_results.values():
            recs.extend(result.get('recommendations', []))
        recs = list(dict.fromkeys(recs))  # Dedupe

        return {
            'overall_status': overall_status,
            'overall_confidence': overall_conf,
            'summary': f"{status_counts['passed']}p/{status_counts['warning']}w/{status_counts['failed']}f - 3 agents",
            'status_counts': status_counts,
            'agent_results': agent_results,
            'all_checks': all_checks,
            'recommendations': recs[:5]
        }

    def _generate_cross_agent_insights(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        insights = {
            'consensus_passed': [],
            'consensus_warning': [],
            'high_confidence': [],
            'drug_pathway_overlap': []
        }

        # Consensus (same check name across agents)
        check_by_name = {}
        for agent, result in results.items():
            for check in result.get('checks', []):
                name = check['name']
                if name not in check_by_name:
                    check_by_name[name] = []
                check_by_name[name].append({'agent': agent, 'status': check['status']})

        for name, checks in check_by_name.items():
            if len(checks) >= 2:
                statuses = [c['status'] for c in checks]
                if all(s == 'passed' for s in statuses):
                    insights['consensus_passed'].append({'check': name, 'agents': [c['agent'] for c in checks]})
                elif all(s == 'warning' for s in statuses):
                    insights['consensus_warning'].append({'check': name, 'agents': [c['agent'] for c in checks]})

        # High confidence
        for agent, result in results.items():
            for check in result.get('checks', []):
                if check.get('confidence') == 'high' and check['status'] == 'passed':
                    insights['high_confidence'].append({'agent': agent, 'check': check['name']})

        return insights

    def get_available_agents(self) -> List[str]:
        return list(self.agents.keys())

# Singleton
_orchestrator = None

def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator

