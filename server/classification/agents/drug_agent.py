
"""
Drug Association Agent
Links biomarkers with therapeutic targets using DGIdb/PPI
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime

from .base_agent import BaseAgent, ValidationResult, ValidationStatus, ConfidenceLevel, ValidationCheck

logger = logging.getLogger(__name__)

class DrugAssociationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Drug Association Agent",
            description="Links biomarkers to therapeutic targets"
        )

    @property
    def system_prompt(self) -> str:
        return """You are a pharmacogenomics expert.

Given biomarkers, identify:
1. Direct drug targets (DGIdb/ChEMBL)
2. Clinical trial evidence
3. Repurposing opportunities
4. Druggability scores

Prioritize FDA-approved + oncology relevance.
Structure: drug name, target, evidence level, cancer relevance."""

from ..utils import fetch_dgidb_drugs_via_graphql\n\n    def fetch_dgidb_drugs(self, biomarkers: List[str]) -> Dict[str, List[Dict]]:
        # \"\"\"Fetch live DGIdb drugs via GraphQL API.\"\"\"
        if not biomarkers:
            return {}
        return fetch_dgidb_drugs_via_graphql(biomarkers)

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        start_time = datetime.now()
        try:
            biomarkers = data.get('biomarkers', [])
            cancer_type = data.get('cancer_type', 'unknown')

            if not biomarkers:
                check = self._create_check("Biomarkers", ValidationStatus.FAILED, "No biomarkers")
                return self._create_result(ValidationStatus.FAILED, "No biomarkers", [check])

            # Fetch
            drug_mapping = self.fetch_dgidb_drugs(biomarkers)
            drug_candidates = []
            for gene, drugs in drug_mapping.items():
                for drug in drugs[:3]:
                    drug_candidates.append({
                        'gene': gene,
                        'drug_name': drug['drug_name'],
                        'score': drug.get('score', 0),
                        'types': drug['types'],
                        'sources': drug['sources']
                    })

            # LLM
            prompt = f"Cancer: {cancer_type}\\nBiomarkers: {', '.join(biomarkers)}\\nCandidates: {len(drug_candidates)}\\nPrioritize."
            reasoning = self._query_llm(prompt)

            checks = [
                self._create_check(
                    name="Drug Lookup",
                    status=ValidationStatus.PASSED if drug_candidates else ValidationStatus.WARNING,
                    message=f"Found {len(drug_candidates)} pairs",
                    evidence={'samples': drug_candidates[:5]}
                ),
                self._create_check("Prioritization", ValidationStatus.PASSED, "Analysis complete", evidence={'reasoning': reasoning})
            ]

            top_drugs = [d['drug_name'] for d in drug_candidates[:5]]
            status = ValidationStatus.PASSED if drug_candidates else ValidationStatus.WARNING
            summary = f"{len(drug_candidates)} candidates (top: {', '.join(top_drugs[:3]) or 'None'})"

            recs = ['Clinical trials for top candidates'] if drug_candidates else ['No targets - expand search']

            return ValidationResult(
                agent_name=self.name,
                overall_status=status,
                overall_confidence=ConfidenceLevel.HIGH if len(drug_candidates) > 3 else ConfidenceLevel.MEDIUM,
                summary=summary,
                checks=checks,
                recommendations=recs,
                metadata={'candidates': drug_candidates},
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            logger.error(f"Drug agent error: {e}")
            return self._create_result(ValidationStatus.ERROR, str(e))

# Singleton
_drug_agent = None
def get_drug_agent() -> 'DrugAssociationAgent':
    global _drug_agent
    if _drug_agent is None:
        _drug_agent = DrugAssociationAgent()
    return _drug_agent

