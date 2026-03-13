"""
Drug Association Agent - Links biomarkers with therapeutic targets
Uses LIVE DrugBank, ClinicalTrials.gov APIs (no mocks)
"""

import json
import time
import logging
from typing import Dict, List, Any

from .base_agent import BaseAgent, ValidationResult, ValidationStatus, ConfidenceLevel, ValidationCheck
from .external_api_client import DrugBankClient, ClinicalTrialsClient, ExternalValidator

logger = logging.getLogger(__name__)


class DrugAssociationAgent(BaseAgent):
    """
    Agent linking biomarkers to therapeutic targets & drugs
    
    Performs LIVE analysis using:
    1. DrugBank: biomarker → drug targets mapping
    2. ClinicalTrials.gov: ongoing trials by drug/biomarker
    3. Approval status verification
    """
    
    def __init__(self):
        super().__init__(
            name="Drug Association Agent",
            description="Links biomarkers with therapeutic targets using DrugBank/ClinicalTrials APIs"
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are a pharmacogenomics expert linking biomarkers to targeted therapies.

Given biomarkers, identify:
1. Direct drug targets (DrugBank verified)
2. Clinical trial status  
3. FDA approval evidence
4. Repurposing potential
5. Drug combination opportunities

Prioritize FDA-approved drugs first, then Phase II/III trials."""
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Link biomarkers to drugs/targets"""
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
            
            # 1. DrugBank target mapping (LIVE)
            drugbank_check = self._validate_drugbank_targets(genes)
            checks.append(drugbank_check)
            
            # 2. Clinical trials analysis (LIVE)
            trials_check = self._validate_clinical_trials(genes, cancer_type)
            checks.append(trials_check)
            
            # 3. Approval status
            approval_check = self._validate_approvals(genes, cancer_type)
            checks.append(approval_check)
            
            # 4. LLM drug reasoning
            llm_check = self._reason_with_llm(genes, cancer_type, checks)
            checks.append(llm_check)
            
            overall_status = self._determine_status(checks)
            
            return self._create_result(
                status=overall_status,
                summary=f"Drug links: {len([c for c in checks if c.status == ValidationStatus.PASSED])} validated pathways",
                checks=checks,
                recommendations=self._get_recommendations(checks),
                metadata={'genes_analyzed': len(genes), 'cancer_type': cancer_type},
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Drug agent error: {e}")
            return self._create_result(
                status=ValidationStatus.ERROR,
                summary=f"Error: {str(e)}",
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def _validate_drugbank_targets(self, genes: List[str]) -> ValidationCheck:
        """LIVE DrugBank target validation"""
        druggable_genes = []
        total_drugs = 0
        
        for gene in genes[:10]:  # Top 10
            # Reverse lookup: gene → drugs (simplified using known patterns)
            targets = DrugBankClient.get_drug_targets(gene.lower())
            if targets:
                druggable_genes.append({
                    'gene': gene,
                    'drugs': [t['target'] for t in targets],
                    'count': len(targets)
                })
                total_drugs += len(targets)
        
        druggable_count = len(druggable_genes)
        
        if druggable_count >= 3:
            return self._create_check(
                name="DrugBank Targets (LIVE)",
                status=ValidationStatus.PASSED,
                message=f"{druggable_count} biomarkers are druggable ({total_drugs} drugs)",
                confidence=ConfidenceLevel.HIGH,
                evidence={'druggable_genes': druggable_genes[:5]}
            )
        elif druggable_count > 0:
            return self._create_check(
                name="DrugBank Targets (LIVE)",
                status=ValidationStatus.WARNING,
                message=f"{druggable_count}/{len(genes)} biomarkers druggable",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'druggable_genes': druggable_genes}
            )
        else:
            return self._create_check(
                name="DrugBank Targets (LIVE)",
                status=ValidationStatus.WARNING,
                message="No direct drug targets found (consider pathway inhibitors)",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'checked_genes': genes[:5]}
            )
    
    def _validate_clinical_trials(self, genes: List[str], cancer_type: str) -> ValidationCheck:
        """LIVE ClinicalTrials.gov analysis"""
        total_trials = 0
        active_trials = 0
        trial_drugs = []
        
        for gene in genes[:5]:
            trials = ClinicalTrialsClient.get_trial_by_drug(gene, cancer_type)
            total_trials += trials.get('total_trials', 0)
            active_trials += trials.get('active_trials', 0)
            if trials.get('total_trials', 0) > 0:
                trial_drugs.append({
                    'gene': gene,
                    'trials': trials['total_trials'],
                    'active': trials['active_trials']
                })
        
        if active_trials >= 2:
            return self._create_check(
                name="Clinical Trials (LIVE)",
                status=ValidationStatus.PASSED,
                message=f"{active_trials} active trials for biomarkers in {cancer_type}",
                confidence=ConfidenceLevel.HIGH,
                evidence={'trial_summary': trial_drugs}
            )
        elif total_trials > 0:
            return self._create_check(
                name="Clinical Trials (LIVE)",
                status=ValidationStatus.PASSED,
                message=f"{total_trials} total trials found",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'trial_summary': trial_drugs}
            )
        else:
            return self._create_check(
                name="Clinical Trials (LIVE)",
                status=ValidationStatus.WARNING,
                message="No clinical trials found for biomarkers",
                confidence=ConfidenceLevel.LOW,
                evidence={'cancer_type': cancer_type}
            )
    
    def _validate_approvals(self, genes: List[str], cancer_type: str) -> ValidationCheck:
        """FDA approval validation"""
        approved = 0
        approved_drugs = []
        
        for gene in genes[:8]:
            approval = DrugBankClient.is_drug_approved(gene, cancer_type)
            if approval.get('approved', False):
                approved += 1
                approved_drugs.append({
                    'gene': gene,
                    'approved_for': approval.get('approved_for', []),
                    'drug': approval.get('drug')
                })
        
        if approved >= 2:
            return self._create_check(
                name="FDA Approvals",
                status=ValidationStatus.PASSED,
                message=f"{approved} FDA-approved drugs for biomarkers",
                confidence=ConfidenceLevel.HIGH,
                evidence={'approved_drugs': approved_drugs}
            )
        elif approved > 0:
            return self._create_check(
                name="FDA Approvals",
                status=ValidationStatus.WARNING,
                message=f"{approved} approved drugs (other indications)",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'approved_drugs': approved_drugs}
            )
        return self._create_check(
            name="FDA Approvals",
            status=ValidationStatus.WARNING,
            message="No FDA-approved drugs found",
            confidence=ConfidenceLevel.LOW
        )
    
    def _reason_with_llm(self, genes: List[str], cancer_type: str, checks: List[ValidationCheck]) -> ValidationCheck:
        """LLM therapeutic reasoning"""
        try:
            evidence = "\n".join([f"{c.name}: {c.message}" for c in checks])
            prompt = f"""Biomarkers: {', '.join(genes[:8])}
Cancer: {cancer_type}

Drug evidence:
{evidence}

Recommend top 3-5 therapeutic strategies with rationale."""
            
            reasoning = self._query_llm(prompt)
            
            return self._create_check(
                name="Therapeutic Reasoning (LLM)",
                status=ValidationStatus.PASSED,
                message="Therapeutic strategy reasoning generated",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'reasoning': reasoning[:400]}
            )
        except Exception:
            return self._create_check(
                name="Therapeutic Reasoning (LLM)",
                status=ValidationStatus.SKIPPED,
                message="LLM unavailable",
                confidence=ConfidenceLevel.NONE
            )
    
    def _determine_status(self, checks: List[ValidationCheck]) -> ValidationStatus:
        if any(c.status == ValidationStatus.ERROR for c in checks): return ValidationStatus.ERROR
        if any(c.status == ValidationStatus.FAILED for c in checks): return ValidationStatus.FAILED
        if any(c.status == ValidationStatus.WARNING for c in checks): return ValidationStatus.WARNING
        return ValidationStatus.PASSED
    
    def _get_recommendations(self, checks: List[ValidationCheck]) -> List[str]:
        recs = ["Prioritize biomarkers with active clinical trials"]
        if any('DrugBank' in c.name and c.status != ValidationStatus.PASSED for c in checks):
            recs.append("Consider pathway inhibitors for non-druggable targets")
        return recs


# Singleton
_drug_assoc_agent = None

def get_drug_association_agent() -> DrugAssociationAgent:
    global _drug_assoc_agent
    if _drug_assoc_agent is None:
        _drug_assoc_agent = DrugAssociationAgent()
    return _drug_assoc_agent

