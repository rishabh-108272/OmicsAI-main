"""
Pathway Reasoning Agent
Performs gene ontology and pathway enrichment analysis on biomarkers
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime

from .base_agent import BaseAgent, ValidationResult, ValidationStatus, ConfidenceLevel, ValidationCheck

logger = logging.getLogger(__name__)

import requests

class PathwayReasoningAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Pathway Reasoning Agent",
            description="Gene Ontology and pathway enrichment analysis"
        )

    def enrichr_pathways(self, biomarkers: List[str]) -> List[Dict]:
        # \"\"\"Query Enrichr API for live pathway enrichment.\"\"\"
        if len(biomarkers) < 2:
            return []
        ENRICHR_URL = "https://maayanlab.cloud/Enrichr/enrich"
        payload = {
            "json": json.dumps({
                "list": biomarkers[:150],  # Max 300, take top
                "background": "Human_Gene_Atlas"
            })
        }
        try:
            resp = requests.post(ENRICHR_URL, data=payload, timeout=10)
            data = resp.json()
            terms = data.get("KEGG_2021_Human", [])[:10]  # Top KEGG pathways
            enriched = []
            for term in terms:
                enriched.append({
                    'pathway': term[0],
                    'description': term[1],
                    'pvalue': float(term[2]),
                    'adj_pval': float(term[6]),
                    'overlap': int(term[3]),
                    'total_genes': int(term[4])
                })
            return enriched
        except Exception:
            logger.warning("Enrichr API failed, fallback empty")
            return []

    @property
    def system_prompt(self) -> str:
        return """You are a bioinformatics expert in pathway analysis.

Analyze input biomarkers:
1. Identify enriched cancer pathways (KEGG/Reactome/GO)
2. Assess biological coherence
3. Provide pathway diagrams/reasoning
4. Rank pathway importance

Output structured pathway enrichment results.
Respond in ValidationCheck format with evidence."""

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        start_time = datetime.now()
        try:
            biomarkers = data.get('biomarkers', [])
            cancer_type = data.get('cancer_type', 'unknown')

            if not biomarkers:
                check = self._create_check(
                    name="Biomarker Input",
                    status=ValidationStatus.FAILED,
                    message="No biomarkers provided"
                )
                return self._create_result(
                    ValidationStatus.FAILED, "No biomarkers", [check]
                )

            checks = []
            enriched_pathways = self.enrichr_pathways(biomarkers)


            # LLM reasoning
            top_pathways = enriched_pathways[:5]
            prompt = f"Cancer: {cancer_type}\\nBiomarkers: {biomarkers}\\nEnriched: {json.dumps(top_pathways, indent=2)}\\nReason pathway coherence."
            reasoning = self._query_llm(prompt)

            # Checks
            checks.append(self._create_check(
                name="Enrichment Analysis",
                status=ValidationStatus.PASSED if enriched_pathways else ValidationStatus.WARNING,
                message=f"Found {len(enriched_pathways)} enriched pathways",
                evidence={'enriched': enriched_pathways[:10]}
            ))
            checks.append(self._create_check(
                name="LLM Reasoning",
                status=ValidationStatus.PASSED,
                message="Pathway reasoning completed",
                evidence={'reasoning': reasoning[:500]}
            ))

            status = ValidationStatus.PASSED if enriched_pathways else ValidationStatus.WARNING
            summary = f"{len(enriched_pathways)} pathways enriched (top: {enriched_pathways[0]['pathway'] if enriched_pathways else 'None'})"
            recommendations = ['Validate top pathways experimentally'] if enriched_pathways else ['Review biomarker list']

            return ValidationResult(
                agent_name=self.name,
                overall_status=status,
                overall_confidence=ConfidenceLevel.HIGH if enriched_pathways else ConfidenceLevel.MEDIUM,
                summary=summary,
                checks=checks,
                recommendations=recommendations,
                metadata={'biomarkers': biomarkers, 'enriched_count': len(enriched_pathways)},
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            logger.error(f"Pathway agent error: {e}")
            return self._create_result(ValidationStatus.ERROR, str(e), error=str(e))

# Singleton
_pathway_agent = None
def get_pathway_agent() -> 'PathwayReasoningAgent':
    global _pathway_agent
    if _pathway_agent is None:
        _pathway_agent = PathwayReasoningAgent()
    return _pathway_agent

