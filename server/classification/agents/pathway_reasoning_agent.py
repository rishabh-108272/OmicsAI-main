"""
Pathway Reasoning Agent - Performs gene ontology and pathway enrichment analysis
Uses LIVE KEGG and UniProt APIs (no mocks)
"""

import json
import time
import logging
from typing import Dict, List, Any
from datetime import datetime

from .base_agent import BaseAgent, ValidationResult, ValidationStatus, ConfidenceLevel, ValidationCheck
from .external_api_client import KEGGClient, UniProtClient, ExternalValidator

logger = logging.getLogger(__name__)


class PathwayReasoningAgent(BaseAgent):
    """
    Agent for gene ontology and pathway enrichment analysis
    
    Performs LIVE analysis using:
    1. KEGG pathway enrichment (cancer pathways: hsa052*)
    2. UniProt GO terms analysis
    3. Pathway relevance scoring for biomarkers
    """
    
    CANCER_PATHWAYS = [
        'hsa05200', 'hsa05210', 'hsa05225', 'hsa05223', 'hsa05213',  # Cancer pathways
        'hsa05219', 'hsa05205', 'hsa05212', 'hsa05214', 'hsa05218',  # More cancer
        'hsa04012', 'hsa04014', 'hsa04110', 'hsa04115', 'hsa04210',  # Signaling
    ]
    
    def __init__(self):
        super().__init__(
            name="Pathway Reasoning Agent",
            description="Performs gene ontology and pathway enrichment analysis using KEGG/UniProt"
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are a bioinformatics pathway analysis expert specializing in cancer biology.

Analyze biomarker pathway enrichment and provide:
1. KEGG pathway relevance to cancer biology
2. Gene Ontology term validation  
3. Biological plausibility assessment
4. Therapeutic pathway implications

Use the LIVE API results provided. Focus on cancer-relevant pathways (hsa052* series)."""
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate biomarkers for pathway enrichment"""
        start_time = time.time()
        try:
            cancer_type = data.get('cancer_type', 'unknown')
            biomarkers = data.get('biomarkers', [])
            
            if not biomarkers:
                return self._create_result(
                    status=ValidationStatus.FAILED,
                    summary="No biomarkers provided",
                    processing_time=time.time() - start_time
                )
            
            genes = [b.get('gene', '').upper() for b in biomarkers if b.get('gene')]
            checks = []
            
            # 1. KEGG pathway enrichment (LIVE)
            kegg_check = self._validate_kegg_enrichment(genes)
            checks.append(kegg_check)
            
            # 2. UniProt GO analysis (LIVE)
            go_check = self._validate_go_terms(genes[:5])  # Top 5
            checks.append(go_check)
            
            # 3. Cancer pathway relevance
            cancer_check = self._validate_cancer_pathways(genes, cancer_type)
            checks.append(cancer_check)
            
            # 4. LLM pathway reasoning
            llm_check = self._reason_with_llm(genes, cancer_type, checks)
            checks.append(llm_check)
            
            overall_status = self._determine_status(checks)
            
            return self._create_result(
                status=overall_status,
                summary=f"Pathway analysis: {len([c for c in checks if c.status == ValidationStatus.PASSED])} passed",
                checks=checks,
                recommendations=self._get_recommendations(checks),
                metadata={'genes_analyzed': len(genes), 'cancer_type': cancer_type},
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Pathway agent error: {e}")
            return self._create_result(
                status=ValidationStatus.ERROR,
                summary=f"Error: {str(e)}",
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def _validate_kegg_enrichment(self, genes: List[str]) -> ValidationCheck:
        """LIVE KEGG pathway enrichment"""
        enriched = []
        for pathway_id in self.CANCER_PATHWAYS[:8]:  # Top 8
            result = KEGGClient.check_pathway_enrichment(genes, pathway_id)
            if result.get('enriched', False):
                enriched.append(result)
        
        if len(enriched) >= 2:
            return self._create_check(
                name="KEGG Enrichment (LIVE)",
                status=ValidationStatus.PASSED,
                message=f"Found {len(enriched)} enriched cancer pathways",
                confidence=ConfidenceLevel.HIGH,
                evidence={'enriched_pathways': enriched[:3]}
            )
        elif enriched:
            return self._create_check(
                name="KEGG Enrichment (LIVE)",
                status=ValidationStatus.WARNING,
                message=f"Minimal enrichment: {len(enriched)} pathways",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'enriched_pathways': enriched}
            )
        else:
            return self._create_check(
                name="KEGG Enrichment (LIVE)",
                status=ValidationStatus.FAILED,
                message="No significant KEGG pathway enrichment",
                confidence=ConfidenceLevel.HIGH,
                evidence={'checked_pathways': self.CANCER_PATHWAYS[:5]}
            )
    
    def _validate_go_terms(self, genes: List[str]) -> ValidationCheck:
        """LIVE UniProt GO term analysis"""
        go_terms = []
        for gene in genes:
            info = UniProtClient.get_protein_info(gene)
            if info and 'functions' in info:
                go_terms.append({'gene': gene, 'functions': info['functions'][:200]})
        
        relevant_terms = sum(1 for gt in go_terms if any(term in gt.get('functions', '').lower() 
                         for term in ['kinase', 'oncogene', 'tumor suppressor', 'apoptosis', 'proliferation']))
        
        if relevant_terms >= 2:
            return self._create_check(
                name="GO Term Analysis (LIVE)",
                status=ValidationStatus.PASSED,
                message=f"{relevant_terms}/{len(go_terms)} genes have cancer-relevant GO terms",
                confidence=ConfidenceLevel.HIGH,
                evidence={'go_terms': go_terms[:3]}
            )
        return self._create_check(
            name="GO Term Analysis (LIVE)",
            status=ValidationStatus.WARNING,
            message=f"Only {relevant_terms}/{len(go_terms)} relevant GO terms",
            confidence=ConfidenceLevel.MEDIUM,
            evidence={'go_terms': go_terms}
        )
    
    def _validate_cancer_pathways(self, genes: List[str], cancer_type: str) -> ValidationCheck:
        """Cancer-specific pathway validation"""
        pathway_result = ExternalValidator.validate_pathway_enrichment(genes)
        enriched_count = pathway_result.get('pathway_count', 0)
        
        status = ValidationStatus.PASSED if enriched_count >= 2 else ValidationStatus.WARNING
        message = f"{enriched_count} cancer pathways enriched for {cancer_type}"
        
        return self._create_check(
            name="Cancer Pathway Relevance",
            status=status,
            message=message,
            confidence=ConfidenceLevel.HIGH if enriched_count >= 3 else ConfidenceLevel.MEDIUM,
            evidence=pathway_result
        )
    
    def _reason_with_llm(self, genes: List[str], cancer_type: str, checks: List[ValidationCheck]) -> ValidationCheck:
        """LLM pathway reasoning"""
        try:
            evidence_summary = "\n".join([f"{c.name}: {c.message}" for c in checks])
            prompt = f"""Biomarkers: {', '.join(genes[:10])}
Cancer: {cancer_type}

LIVE API results:
{evidence_summary}

Provide biological reasoning and therapeutic implications."""
            
            reasoning = self._query_llm(prompt, temperature=0.3)
            
            return self._create_check(
                name="Pathway Reasoning (LLM)",
                status=ValidationStatus.PASSED,
                message="Biological pathway reasoning completed",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'reasoning': reasoning[:500]}
            )
        except Exception:
            return self._create_check(
                name="Pathway Reasoning (LLM)", 
                status=ValidationStatus.SKIPPED,
                message="LLM reasoning unavailable",
                confidence=ConfidenceLevel.NONE
            )
    
    def _determine_status(self, checks: List[ValidationCheck]) -> ValidationStatus:
        if any(c.status == ValidationStatus.ERROR for c in checks):
            return ValidationStatus.ERROR
        if any(c.status == ValidationStatus.FAILED for c in checks):
            return ValidationStatus.FAILED
        if any(c.status == ValidationStatus.WARNING for c in checks):
            return ValidationStatus.WARNING
        return ValidationStatus.PASSED
    
    def _get_recommendations(self, checks: List[ValidationCheck]) -> List[str]:
        recs = []
        if any(c.status == ValidationStatus.FAILED for c in checks if 'KEGG' in c.name):
            recs.append("Run GSEA analysis for deeper pathway insights")
        recs.append("Prioritize biomarkers from enriched cancer pathways")
        return recs


# Singleton
_pathway_agent = None

def get_pathway_reasoning_agent() -> PathwayReasoningAgent:
    global _pathway_agent
    if _pathway_agent is None:
        _pathway_agent = PathwayReasoningAgent()
    return _pathway_agent

