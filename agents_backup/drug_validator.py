"""
Drug Repurposing Validation Agent
Validates drug-biomarker interactions and repurposing candidates
"""

import json
import time
import logging
from typing import Any, Dict, List
from datetime import datetime

from .base_agent import (
    BaseAgent, 
    ValidationResult, 
    ValidationStatus, 
    ConfidenceLevel,
    ValidationCheck
)
from .external_api_client import (
    DrugBankClient,
    ClinicalTrialsClient,
    validate_drug_external
)

logger = logging.getLogger(__name__)


class DrugRepurposingValidator(BaseAgent):
    """
    Agent that validates drug repurposing candidates
    
    Performs:
    1. Drug-gene interaction verification
    2. Mechanism of action validation
    3. Clinical trial evidence check
    4. Safety and efficacy assessment
    5. Network proximity analysis
    """
    
    # Known drug targets by pathway
    DRUG_TARGETS = {
        'PI3K_AKT': [
            {'drug': 'Everolimus', 'target': 'MTOR', 'approval': 'FDA', 'cancer_types': [' RCC', 'Breast']},
            {'drug': 'Sirolimus', 'target': 'MTOR', 'approval': 'FDA', 'cancer_types': ['RCC']},
            {'drug': 'Alpelisib', 'target': 'PIK3CA', 'approval': 'FDA', 'cancer_types': ['Breast']},
            {'drug': 'Copanlisib', 'target': 'PIK3CA', 'approval': 'FDA', 'cancer_types': ['Lymphoma']},
            {'drug': 'Temsirolimus', 'target': 'MTOR', 'approval': 'FDA', 'cancer_types': ['RCC']},
        ],
        'MAPK': [
            {'drug': 'Vemurafenib', 'target': 'BRAF', 'approval': 'FDA', 'cancer_types': ['Melanoma', 'CRC']},
            {'drug': 'Dabrafenib', 'target': 'BRAF', 'approval': 'FDA', 'cancer_types': ['Melanoma', 'Lung']},
            {'drug': 'Trametinib', 'target': 'MAP2K1', 'approval': 'FDA', 'cancer_types': ['Melanoma', 'Lung']},
            {'drug': 'Cobimetinib', 'target': 'MAP2K1', 'approval': 'FDA', 'cancer_types': ['Melanoma']},
            {'drug': 'Selumetinib', 'target': 'MAP2K1', 'approval': 'Clinical', 'cancer_types': ['Thyroid']},
        ],
        'EGFR': [
            {'drug': 'Erlotinib', 'target': 'EGFR', 'approval': 'FDA', 'cancer_types': ['Lung', 'Pancreatic']},
            {'drug': 'Gefitinib', 'target': 'EGFR', 'approval': 'FDA', 'cancer_types': ['Lung']},
            {'drug': 'Osimertinib', 'target': 'EGFR', 'approval': 'FDA', 'cancer_types': ['Lung']},
            {'drug': 'Cetuximab', 'target': 'EGFR', 'approval': 'FDA', 'cancer_types': ['CRC', 'Head & Neck']},
            {'drug': 'Panitumumab', 'target': 'EGFR', 'approval': 'FDA', 'cancer_types': ['CRC']},
        ],
        'ALK': [
            {'drug': 'Crizotinib', 'target': 'ALK', 'approval': 'FDA', 'cancer_types': ['Lung']},
            {'drug': 'Alectinib', 'target': 'ALK', 'approval': 'FDA', 'cancer_types': ['Lung']},
            {'drug': 'Ceritinib', 'target': 'ALK', 'approval': 'FDA', 'cancer_types': ['Lung']},
            {'drug': 'Lorlatinib', 'target': 'ALK', 'approval': 'FDA', 'cancer_types': ['Lung']},
        ],
        'BRCA': [
            {'drug': 'Olaparib', 'target': 'PARP1', 'approval': 'FDA', 'cancer_types': ['Ovarian', 'Breast', 'Prostate']},
            {'drug': 'Niraparib', 'target': 'PARP1', 'approval': 'FDA', 'cancer_types': ['Ovarian']},
            {'drug': 'Rucaparib', 'target': 'PARP1', 'approval': 'FDA', 'cancer_types': ['Ovarian', 'Prostate']},
            {'drug': 'Talazoparib', 'target': 'PARP1', 'approval': 'FDA', 'cancer_types': ['Breast']},
        ],
        'VEGF': [
            {'drug': 'Bevacizumab', 'target': 'VEGFA', 'approval': 'FDA', 'cancer_types': ['CRC', 'Lung', 'RCC', 'Ovarian']},
            {'drug': 'Sunitinib', 'target': 'VEGFR2', 'approval': 'FDA', 'cancer_types': ['RCC', 'GIST']},
            {'drug': 'Sorafenib', 'target': 'VEGFR2', 'approval': 'FDA', 'cancer_types': ['HCC', 'RCC']},
            {'drug': 'Pazopanib', 'target': 'VEGFR2', 'approval': 'FDA', 'cancer_types': ['RCC']},
            {'drug': 'Axitinib', 'target': 'VEGFR2', 'approval': 'FDA', 'cancer_types': ['RCC']},
        ],
        'JAK_STAT': [
            {'drug': 'Ruxolitinib', 'target': 'JAK2', 'approval': 'FDA', 'cancer_types': ['Myelofibrosis']},
            {'drug': 'Fedratinib', 'target': 'JAK2', 'approval': 'FDA', 'cancer_types': ['Myelofibrosis']},
        ],
        'CDK4_6': [
            {'drug': 'Palbociclib', 'target': 'CDK4', 'approval': 'FDA', 'cancer_types': ['Breast']},
            {'drug': 'Ribociclib', 'target': 'CDK4', 'approval': 'FDA', 'cancer_types': ['Breast']},
            {'drug': 'Abemaciclib', 'target': 'CDK4', 'approval': 'FDA', 'cancer_types': ['Breast']},
        ],
    }
    
    # Clinical trial phases
    TRIAL_PHASES = ['Phase 0', 'Phase I', 'Phase II', 'Phase III', 'Phase IV', 'Approved']
    
    def __init__(self):
        super().__init__(
            name="Drug Repurposing Validator",
            description="Validates drug-biomarker interactions for repurposing"
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are a pharmacogenomics expert specializing in drug repurposing and molecular therapy.

Your role is to VALIDATE drug repurposing candidates from network analysis. You are NOT a chatbot - you are a reasoning module that:

1. Verifies drug-gene interactions against known databases
2. Validates mechanism of action (MoA) plausibility
3. Checks clinical trial evidence and approval status
4. Assesses network proximity and connectivity
5. Evaluates safety and efficacy evidence

For each drug candidate, you must:
- Verify the target exists and is relevant
- Check approval status and clinical trial phase
- Assess network proximity to seed biomarkers
- Provide evidence-based recommendations

Focus on:
- Is the drug target biologically relevant to the cancer?
- What is the evidence level (FDA approved, clinical trial, preclinical)?
- Are there contraindications or safety concerns?
- What is the strength of network evidence?"""
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate drug repurposing results
        
        Expected data format:
        {
            'cancer_type': str,
            'biomarkers': list,  # Seed biomarkers
            'candidates': [
                {
                    'drug_name': str,
                    'target': str,
                    'hops_from_biomarker': int,
                    'score': float,
                    'evidence': str
                }
            ],
            'graph_data': dict  # Network analysis data
        }
        """
        import time
        start_time = time.time()
        
        try:
            # Extract data
            cancer_type = data.get('cancer_type', 'unknown')
            biomarkers = data.get('biomarkers', [])
            candidates = data.get('candidates', [])
            graph_data = data.get('graph_data', {})
            
            checks = []
            recommendations = []
            
            if not candidates:
                checks.append(self._create_check(
                    name="Drug Candidates",
                    status=ValidationStatus.FAILED,
                    message="No drug candidates provided for validation",
                    confidence=ConfidenceLevel.NONE
                ))
                
                return self._create_result(
                    status=ValidationStatus.FAILED,
                    summary="No drug candidates to validate",
                    checks=checks,
                    processing_time=time.time() - start_time
                )
            
            # 1. Validate candidate count
            checks.append(self._validate_candidate_count(candidates))
            
            # 2. Validate known drug targets
            target_check = self._validate_known_targets(candidates)
            checks.append(target_check)
            
            # 3. Validate network proximity
            proximity_check = self._validate_network_proximity(candidates, biomarkers)
            checks.append(proximity_check)
            
            # 4. Validate clinical evidence
            evidence_check = self._validate_clinical_evidence(candidates, cancer_type)
            checks.append(evidence_check)
            
            # 5. Validate pathway alignment
            pathway_check = self._validate_pathway_alignment(candidates, biomarkers)
            checks.append(pathway_check)
            
            # 6. External API validation (DrugBank, ClinicalTrials.gov)
            external_check = self._validate_with_external_apis(candidates, cancer_type, biomarkers)
            checks.append(external_check)
            
            # 7. LLM deep validation
            llm_check = self._validate_with_llm(candidates, cancer_type, biomarkers)
            checks.append(llm_check)
            
            # Determine overall status
            overall_status = self._determine_overall_status(checks)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(checks, candidates, cancer_type)
            
            processing_time = time.time() - start_time
            
            return ValidationResult(
                agent_name=self.name,
                overall_status=overall_status,
                overall_confidence=self._calculate_overall_confidence(checks),
                summary=self._generate_summary(checks, overall_status),
                checks=checks,
                recommendations=recommendations,
                metadata={
                    'cancer_type': cancer_type,
                    'candidate_count': len(candidates),
                    'seed_biomarkers': biomarkers,
                    'validation_timestamp': datetime.now().isoformat()
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Drug repurposing validation error: {e}")
            processing_time = time.time() - start_time
            
            return self._create_result(
                status=ValidationStatus.ERROR,
                summary=f"Validation error: {str(e)}",
                processing_time=processing_time,
                error=str(e)
            )
    
    def _validate_candidate_count(self, candidates: List[Dict]) -> ValidationCheck:
        """Validate if sufficient candidates were found"""
        count = len(candidates)
        
        if count < 3:
            return self._create_check(
                name="Candidate Count",
                status=ValidationStatus.WARNING,
                message=f"Only {count} drug candidates found. Consider expanding network.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'count': count}
            )
        elif count > 50:
            return self._create_check(
                name="Candidate Count",
                status=ValidationStatus.WARNING,
                message=f"Large number of candidates ({count}). Prioritize by evidence.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'count': count}
            )
        else:
            return self._create_check(
                name="Candidate Count",
                status=ValidationStatus.PASSED,
                message=f"Found {count} drug candidates for validation",
                confidence=ConfidenceLevel.HIGH,
                evidence={'count': count}
            )
    
    def _validate_known_targets(self, candidates: List[Dict]) -> ValidationCheck:
        """Validate against known drug targets"""
        known_drugs = {}
        for pathway, drugs in self.DRUG_TARGETS.items():
            for drug in drugs:
                known_drugs[drug['target'].upper()] = drug
        
        validated = []
        unknown = []
        
        for candidate in candidates:
            target = candidate.get('target', '')
            if target.upper() in known_drugs:
                validated.append({
                    'drug': candidate.get('drug_name'),
                    'target': target,
                    'known_info': known_drugs[target.upper()]
                })
            else:
                unknown.append(candidate.get('drug_name'))
        
        if not validated:
            return self._create_check(
                name="Known Drug Targets",
                status=ValidationStatus.WARNING,
                message="No validated drug targets found. All candidates are novel.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={
                    'validated_count': 0,
                    'unknown_count': len(unknown)
                }
            )
        
        return self._create_check(
            name="Known Drug Targets",
            status=ValidationStatus.PASSED,
            message=f"Found {len(validated)} candidates with known drug targets",
            confidence=ConfidenceLevel.HIGH,
            evidence={
                'validated_drugs': validated,
                'validated_count': len(validated),
                'unknown_count': len(unknown)
            }
        )
    
    def _validate_network_proximity(
        self, 
        candidates: List[Dict], 
        biomarkers: List[str]
    ) -> ValidationCheck:
        """Validate network proximity to seed biomarkers"""
        if not candidates:
            return self._create_check(
                name="Network Proximity",
                status=ValidationStatus.FAILED,
                message="No candidates to validate",
                confidence=ConfidenceLevel.NONE
            )
        
        # Check hops distribution
        hops = [c.get('hops_from_biomarker', 999) for c in candidates]
        scores = [c.get('score', 0) for c in candidates]
        
        direct_targets = sum(1 for h in hops if h <= 1)
        nearby_targets = sum(1 for h in hops if h <= 2)
        
        if direct_targets == 0:
            return self._create_check(
                name="Network Proximity",
                status=ValidationStatus.WARNING,
                message="No direct drug targets found. Consider closer biomarkers.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={
                    'direct_targets': 0,
                    'nearby_targets': nearby_targets,
                    'max_hops': max(hops) if hops else None
                }
            )
        
        avg_hops = sum(hops) / len(hops) if hops else 999
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return self._create_check(
            name="Network Proximity",
            status=ValidationStatus.PASSED,
            message=f"{direct_targets} direct targets, avg {avg_hops:.1f} hops from biomarkers",
            confidence=ConfidenceLevel.HIGH if direct_targets >= 2 else ConfidenceLevel.MEDIUM,
            evidence={
                'direct_targets': direct_targets,
                'nearby_targets': nearby_targets,
                'avg_hops': avg_hops,
                'avg_score': avg_score
            }
        )
    
    def _validate_clinical_evidence(
        self, 
        candidates: List[Dict], 
        cancer_type: str
    ) -> ValidationCheck:
        """Validate clinical evidence for candidates"""
        approval_status = {'FDA': 0, 'Clinical': 0, 'Preclinical': 0, 'Unknown': 0}
        
        # Check known drugs
        known_drugs = {}
        for pathway, drugs in self.DRUG_TARGETS.items():
            for drug in drugs:
                known_drugs[drug['drug'].upper()] = drug
        
        evidence_levels = []
        
        for candidate in candidates:
            drug_name = candidate.get('drug_name', '')
            
            if drug_name.upper() in known_drugs:
                info = known_drugs[drug_name.upper()]
                approval = info.get('approval', 'Unknown')
                
                # Check if approved for this cancer type
                cancer_types = info.get('cancer_types', [])
                if any(cancer_type.lower() in ct.lower() for ct in cancer_types):
                    approval_status['FDA'] = approval_status.get('FDA', 0) + 1
                    evidence_levels.append({
                        'drug': drug_name,
                        'level': 'FDA Approved',
                        'cancer_type': cancer_type
                    })
                else:
                    approval_status['Clinical'] = approval_status.get('Clinical', 0) + 1
                    evidence_levels.append({
                        'drug': drug_name,
                        'level': 'Approved for other cancers',
                        'cancer_type': cancer_type
                    })
            else:
                approval_status['Unknown'] = approval_status.get('Unknown', 0) + 1
        
        fda_count = approval_status.get('FDA', 0)
        
        if fda_count > 0:
            return self._create_check(
                name="Clinical Evidence",
                status=ValidationStatus.PASSED,
                message=f"Found {fda_count} FDA-approved drugs for this cancer type",
                confidence=ConfidenceLevel.HIGH,
                evidence={
                    'approval_status': approval_status,
                    'evidence_levels': evidence_levels
                }
            )
        elif approval_status.get('Clinical', 0) > 0:
            return self._create_check(
                name="Clinical Evidence",
                status=ValidationStatus.PASSED,
                message="Found drugs approved for other cancers (repurposing candidates)",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={
                    'approval_status': approval_status,
                    'evidence_levels': evidence_levels
                }
            )
        else:
            return self._create_check(
                name="Clinical Evidence",
                status=ValidationStatus.WARNING,
                message="Limited clinical evidence. Consider experimental therapies.",
                confidence=ConfidenceLevel.LOW,
                evidence={'approval_status': approval_status}
            )
    
    def _validate_pathway_alignment(
        self, 
        candidates: List[Dict], 
        biomarkers: List[str]
    ) -> ValidationCheck:
        """Validate pathway alignment of drug targets"""
        biomarker_set = set([b.upper() for b in biomarkers])
        
        pathway_coverage = {}
        for pathway, drugs in self.DRUG_TARGETS.items():
            for drug in drugs:
                target = drug['target'].upper()
                if target in biomarker_set:
                    if pathway not in pathway_coverage:
                        pathway_coverage[pathway] = []
                    pathway_coverage[pathway].append({
                        'drug': drug['drug'],
                        'target': target
                    })
        
        if not pathway_coverage:
            return self._create_check(
                name="Pathway Alignment",
                status=ValidationStatus.WARNING,
                message="Drug targets not aligned with known pathways",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'pathway_coverage': {}}
            )
        
        return self._create_check(
            name="Pathway Alignment",
            status=ValidationStatus.PASSED,
            message=f"Drug targets aligned with {len(pathway_coverage)} cancer-related pathways",
            confidence=ConfidenceLevel.HIGH,
            evidence={
                'pathway_coverage': pathway_coverage,
                'pathway_count': len(pathway_coverage)
            }
        )
    
    def _validate_with_external_apis(
        self, 
        candidates: List[Dict], 
        cancer_type: str,
        biomarkers: List[str]
    ) -> ValidationCheck:
        """Validate using external APIs (DrugBank, ClinicalTrials.gov)"""
        try:
            # Get top candidate drugs to validate
            top_candidates = sorted(
                candidates, 
                key=lambda x: x.get('score', 0), 
                reverse=True
            )[:5]
            
            validated_drugs = []
            trials_info = []
            
            for candidate in top_candidates:
                drug_name = candidate.get('drug_name', '')
                target = candidate.get('target', '')
                
                # Validate drug-target relationship
                target_validation = DrugBankClient.get_drug_targets(drug_name)
                
                # Check approval status
                approval = DrugBankClient.is_drug_approved(drug_name, cancer_type)
                
                # Get clinical trials
                trials = ClinicalTrialsClient.get_trial_by_drug(drug_name, cancer_type)
                
                validated_drugs.append({
                    'drug': drug_name,
                    'target': target,
                    'approved': approval.get('approved', False),
                    'approved_for': approval.get('approved_for', []),
                    'clinical_trials': trials.get('total_trials', 0),
                    'active_trials': trials.get('active_trials', 0)
                })
                
                if trials.get('total_trials', 0) > 0:
                    trials_info.append({
                        'drug': drug_name,
                        'trials': trials.get('total_trials', 0),
                        'active': trials.get('active_trials', 0)
                    })
            
            # Count FDA approved drugs
            approved_count = sum(1 for d in validated_drugs if d['approved'])
            trials_count = len(trials_info)
            
            if approved_count > 0:
                return self._create_check(
                    name="External Database Validation",
                    status=ValidationStatus.PASSED,
                    message=f"Found {approved_count} FDA-approved drugs, {trials_count} with active clinical trials",
                    confidence=ConfidenceLevel.HIGH,
                    evidence={
                        'source': 'drugbank_clinicaltrials',
                        'validated_drugs': validated_drugs,
                        'approved_count': approved_count,
                        'trials_info': trials_info
                    }
                )
            elif trials_count > 0:
                return self._create_check(
                    name="External Database Validation",
                    status=ValidationStatus.PASSED,
                    message=f"Found {trials_count} drugs with ongoing clinical trials",
                    confidence=ConfidenceLevel.MEDIUM,
                    evidence={
                        'source': 'clinicaltrials',
                        'trials_info': trials_info
                    }
                )
            else:
                return self._create_check(
                    name="External Database Validation",
                    status=ValidationStatus.WARNING,
                    message="Limited external validation data available",
                    confidence=ConfidenceLevel.LOW,
                    evidence={
                        'source': 'none',
                        'validated_drugs': validated_drugs
                    }
                )
                
        except Exception as e:
            logger.warning(f"External API validation failed: {e}")
            return self._create_check(
                name="External Database Validation",
                status=ValidationStatus.SKIPPED,
                message=f"External validation skipped: {str(e)}",
                confidence=ConfidenceLevel.NONE
            )
    
    def _validate_with_llm(
        self, 
        candidates: List[Dict], 
        cancer_type: str,
        biomarkers: List[str]
    ) -> ValidationCheck:
        """Use LLM for deep validation"""
        try:
            # Get top candidates
            top_candidates = sorted(
                candidates, 
                key=lambda x: x.get('score', 0), 
                reverse=True
            )[:8]
            
            candidate_summary = []
            for c in top_candidates:
                drug = c.get('drug_name', 'Unknown')
                target = c.get('target', 'Unknown')
                hops = c.get('hops_from_biomarker', 'N/A')
                score = c.get('score', 0)
                candidate_summary.append(f"{drug}: target={target}, hops={hops}, score={score:.3f}")
            
            prompt = f"""Analyze these drug repurposing candidates for {cancer_type}:

Seed Biomarkers: {', '.join(biomarkers[:10])}

Candidates:
{chr(10).join(candidate_summary)}

For each candidate, assess:
1. Mechanistic plausibility (does targeting this gene make biological sense?)
2. Clinical potential (any existing evidence?)
3. Network quality (is the proximity compelling?)

Provide JSON response:
{{
    "overall_assessment": "brief summary",
    "top_recommendations": ["list of top 3-5 drugs with reasoning"],
    "mechanistic_plausibility": ["drugs with strong mechanistic support"],
    "network_quality": "good/acceptable/poor",
    "clinical_potential": "high/medium/low",
    "concerns": ["any concerns"],
    "confidence": "high/medium/low"
}}"""

            response = self._query_llm_structured(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "overall_assessment": {"type": "string"},
                        "top_recommendations": {"type": "array", "items": {"type": "string"}},
                        "mechanistic_plausibility": {"type": "array", "items": {"type": "string"}},
                        "network_quality": {"type": "string", "enum": ["good", "acceptable", "poor"]},
                        "clinical_potential": {"type": "string", "enum": ["high", "medium", "low"]},
                        "concerns": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
                    }
                }
            )
            
            conf_map = {
                'high': ConfidenceLevel.HIGH,
                'medium': ConfidenceLevel.MEDIUM,
                'low': ConfidenceLevel.LOW
            }
            
            status_map = {
                'good': ValidationStatus.PASSED,
                'acceptable': ValidationStatus.PASSED,
                'poor': ValidationStatus.FAILED
            }
            
            status = status_map.get(response.get('network_quality'), ValidationStatus.PASSED)
            
            return self._create_check(
                name="LLM Deep Validation",
                status=status,
                message=response.get('overall_assessment', 'LLM validation completed'),
                confidence=conf_map.get(response.get('confidence', 'medium'), ConfidenceLevel.MEDIUM),
                evidence={
                    'top_recommendations': response.get('top_recommendations', []),
                    'mechanistic_plausibility': response.get('mechanistic_plausibility', []),
                    'network_quality': response.get('network_quality'),
                    'clinical_potential': response.get('clinical_potential'),
                    'concerns': response.get('concerns', [])
                }
            )
            
        except Exception as e:
            logger.warning(f"LLM drug validation failed: {e}")
            return self._create_check(
                name="LLM Deep Validation",
                status=ValidationStatus.SKIPPED,
                message=f"LLM validation skipped: {str(e)}",
                confidence=ConfidenceLevel.NONE
            )
    
    def _determine_overall_status(self, checks: List[ValidationCheck]) -> ValidationStatus:
        """Determine overall status"""
        if any(c.status == ValidationStatus.ERROR for c in checks):
            return ValidationStatus.ERROR
        if any(c.status == ValidationStatus.FAILED for c in checks):
            return ValidationStatus.FAILED
        if any(c.status == ValidationStatus.WARNING for c in checks):
            return ValidationStatus.WARNING
        return ValidationStatus.PASSED
    
    def _calculate_overall_confidence(self, checks: List[ValidationCheck]) -> ConfidenceLevel:
        """Calculate overall confidence"""
        if not checks:
            return ConfidenceLevel.NONE
        
        high_count = sum(1 for c in checks if c.confidence == ConfidenceLevel.HIGH)
        total = len(checks)
        
        if high_count / total >= 0.6:
            return ConfidenceLevel.HIGH
        elif high_count / total >= 0.3:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _generate_recommendations(
        self, 
        checks: List[ValidationCheck], 
        candidates: List[Dict],
        cancer_type: str
    ) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        # Get evidence check
        for check in checks:
            if check.status == ValidationStatus.FAILED:
                if 'Target' in check.name:
                    recommendations.append(
                        "Consider experimental drugs or novel targets"
                    )
                elif 'Proximity' in check.name:
                    recommendations.append(
                        "Expand network analysis to find closer drug targets"
                    )
            elif check.status == ValidationStatus.WARNING:
                if 'Evidence' in check.name:
                    recommendations.append(
                        "Look for ongoing clinical trials for repurposing"
                    )
        
        # Add specific recommendations for high-scoring candidates
        top_candidates = sorted(
            candidates, 
            key=lambda x: x.get('score', 0), 
            reverse=True
        )[:3]
        
        for c in top_candidates:
            recommendations.append(
                f"Consider {c.get('drug_name')} (score: {c.get('score', 0):.3f})"
            )
        
        return recommendations
    
    def _generate_summary(self, checks: List[ValidationCheck], status: ValidationStatus) -> str:
        """Generate summary"""
        passed = sum(1 for c in checks if c.status == ValidationStatus.PASSED)
        failed = sum(1 for c in checks if c.status == ValidationStatus.FAILED)
        warnings = sum(1 for c in checks if c.status == ValidationStatus.WARNING)
        
        return f"Drug validation {status.value}: {passed} passed, {warnings} warnings, {failed} failed"


# Singleton instance
_drug_validator = None

def get_drug_validator() -> DrugRepurposingValidator:
    """Get singleton DrugRepurposingValidator instance"""
    global _drug_validator
    if _drug_validator is None:
        _drug_validator = DrugRepurposingValidator()
    return _drug_validator

