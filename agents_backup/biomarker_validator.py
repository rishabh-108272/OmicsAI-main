"""
Biomarker Discovery Validation Agent
Validates discovered biomarkers against known pathways and literature
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

logger = logging.getLogger(__name__)


class BiomarkerValidator(BaseAgent):
    """
    Agent that validates discovered biomarkers
    
    Performs:
    1. Cross-validation with KEGG/Reactome pathways
    2. Literature evidence verification
    3. Statistical significance assessment
    4. Pathway enrichment analysis
    """
    
    # Known cancer-related pathways
    CANCER_PATHWAYS = {
        'PI3K_AKT': ['PIK3CA', 'AKT1', 'AKT2', 'MTOR', 'PTEN', 'TSC1', 'TSC2'],
        'MAPK': ['KRAS', 'NRAS', 'BRAF', 'RAF1', 'MAP2K1', 'MAP2K2', 'ERK1', 'ERK2'],
        'WNT': ['CTNNB1', 'APC', 'AXIN1', 'AXIN2', 'GSK3B', 'MYC'],
        'P53': ['TP53', 'MDM2', 'CDKN1A', 'BCL2', 'BAX'],
        'CELL_CYCLE': ['CDK4', 'CDK6', 'CDK2', 'CCND1', 'RB1', 'E2F1'],
        'JAK_STAT': ['JAK1', 'JAK2', 'STAT3', 'STAT5', 'IL6'],
        'NOTCH': ['NOTCH1', 'NOTCH2', 'NOTCH3', 'JAG1', 'HES1'],
        'HEDGEHOG': ['SMO', 'GLI1', 'GLI2', 'PTCH1', 'SHH'],
        'TGF_BETA': ['TGFBR1', 'TGFBR2', 'SMAD2', 'SMAD3', 'SMAD4'],
        'HIPPO': ['YAP1', 'WWTR1', 'TEAD1', 'LATS1', 'LATS2'],
        'METABOLISM': ['HK2', 'PKM2', 'LDHA', 'GLS1', 'GOT2'],
        'DNA_REPAIR': ['BRCA1', 'BRCA2', 'MLH1', 'MSH2', 'ATM', 'ATR'],
        'APOPTOSIS': ['BCL2', 'BCL2L1', 'BAX', 'CASP3', 'CASP9', 'PMAIP1'],
        'EMT': ['SNAI1', 'SNAI2', 'TWIST1', 'ZEB1', 'VIM', 'CDH2'],
        'ANGIOGENESIS': ['VEGFA', 'VEGFR1', 'VEGFR2', 'ANGPT1', 'ANGPT2', 'HIF1A'],
        'IMMUNE': ['PDL1', 'PD1', 'CTLA4', 'IL2', 'IFNG', 'TNF'],
    }
    
    # Known disease associations
    DISEASE_BIOMARKERS = {
        'colorectal_cancer': ['KRAS', 'NRAS', 'BRAF', 'APC', 'TP53', 'SMAD4', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'EGFR', 'MET', 'ERBB2'],
        'liver_cancer': ['TP53', 'CTNNB1', 'AXIN1', 'ARID1A', 'ARID2', 'TERT', 'RB1', 'CDKN2A', 'IGF2', 'AFP', 'GPC3', 'SERPINA1'],
        'lung_cancer': ['EGFR', 'ALK', 'ROS1', 'KRAS', 'BRAF', 'MET', 'ERBB2', 'TP53', 'RB1', 'PTEN', 'STK11', 'KEAP1', 'NFE2L2'],
        'breast_cancer': ['ESR1', 'PGR', 'ERBB2', 'BRCA1', 'BRCA2', 'TP53', 'PIK3CA', 'CDH1', 'GATA3'],
        'prostate_cancer': ['TMPRSS2', 'ERG', 'PTEN', 'TP53', 'RB1', 'SPOP', 'CHD1'],
        'pancreatic_cancer': ['KRAS', 'CDKN2A', 'TP53', 'SMAD4', 'BRCA2', 'ATM'],
    }
    
    def __init__(self):
        super().__init__(
            name="Biomarker Validator",
            description="Validates discovered biomarkers against pathways and literature"
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are a computational biology expert specializing in biomarker validation.

Your role is to VALIDATE discovered biomarkers from XAI analysis. You are NOT a chatbot - you are a reasoning module that:

1. Cross-validates biomarkers against known biological pathways
2. Verifies statistical significance and effect sizes
3. Checks for literature evidence supporting biomarker-disease associations
4. Assesses pathway enrichment and biological relevance
5. Identifies potential false positives or artifacts

For each biomarker, you must:
- Verify involvement in cancer-related pathways
- Check statistical significance (p-value, effect size)
- Provide literature-based evidence
- Assign confidence level

Focus on:
- Is the biomarker functionally relevant to the cancer type?
- Are there known drug targets for this biomarker?
- Is the statistical evidence robust?"""
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate biomarker discovery results
        
        Expected data format:
        {
            'cancer_type': str,
            'biomarkers': [
                {'gene': str, 'importance': float, 'p_value': float, ...}
            ],
            'pathway_data': dict,  # Optional pathway enrichment
            'heatmap_data': dict  # Optional gene expression heatmap
        }
        """
        import time
        start_time = time.time()
        
        try:
            # Extract data
            cancer_type = data.get('cancer_type', 'unknown')
            biomarkers = data.get('biomarkers', [])
            pathway_data = data.get('pathway_data', {})
            heatmap_data = data.get('heatmap_data', {})
            
            checks = []
            recommendations = []
            
            if not biomarkers:
                checks.append(self._create_check(
                    name="Biomarker List",
                    status=ValidationStatus.FAILED,
                    message="No biomarkers provided for validation",
                    confidence=ConfidenceLevel.NONE
                ))
                
                return self._create_result(
                    status=ValidationStatus.FAILED,
                    summary="No biomarkers to validate",
                    checks=checks,
                    processing_time=time.time() - start_time
                )
            
            # 1. Validate biomarker count
            checks.append(self._validate_biomarker_count(biomarkers))
            
            # 2. Validate pathway enrichment
            pathway_check = self._validate_pathway_enrichment(biomarkers, pathway_data)
            checks.append(pathway_check)
            
            # 3. Validate against known disease biomarkers
            disease_check = self._validate_disease_association(biomarkers, cancer_type)
            checks.append(disease_check)
            
            # 4. Validate statistical significance
            stats_check = self._validate_statistical_significance(biomarkers)
            checks.append(stats_check)
            
            # 5. Check for pathway diversity
            diversity_check = self._validate_pathway_diversity(biomarkers)
            checks.append(diversity_check)
            
            # 6. LLM-based deep validation
            llm_check = self._validate_with_llm(biomarkers, cancer_type)
            checks.append(llm_check)
            
            # Determine overall status
            overall_status = self._determine_overall_status(checks)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(checks, cancer_type, biomarkers)
            
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
                    'biomarker_count': len(biomarkers),
                    'validation_timestamp': datetime.now().isoformat()
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Biomarker validation error: {e}")
            processing_time = time.time() - start_time
            
            return self._create_result(
                status=ValidationStatus.ERROR,
                summary=f"Validation error: {str(e)}",
                processing_time=processing_time,
                error=str(e)
            )
    
    def _validate_biomarker_count(self, biomarkers: List[Dict]) -> ValidationCheck:
        """Validate if sufficient biomarkers were discovered"""
        count = len(biomarkers)
        
        if count < 5:
            return self._create_check(
                name="Biomarker Count",
                status=ValidationStatus.WARNING,
                message=f"Only {count} biomarkers discovered. Consider lowering significance threshold.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'count': count}
            )
        elif count > 100:
            return self._create_check(
                name="Biomarker Count",
                status=ValidationStatus.WARNING,
                message=f"Large number of biomarkers ({count}). May include false positives.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'count': count}
            )
        else:
            return self._create_check(
                name="Biomarker Count",
                status=ValidationStatus.PASSED,
                message=f"Appropriate number of biomarkers ({count}) discovered",
                confidence=ConfidenceLevel.HIGH,
                evidence={'count': count}
            )
    
    def _validate_pathway_enrichment(
        self, 
        biomarkers: List[Dict], 
        pathway_data: Dict
    ) -> ValidationCheck:
        """Validate pathway enrichment"""
        biomarker_genes = [b.get('gene', '').upper() for b in biomarkers]
        
        # Find enriched pathways
        enriched_pathways = []
        for pathway_name, pathway_genes in self.CANCER_PATHWAYS.items():
            overlap = set(biomarker_genes) & set(pathway_genes)
            if len(overlap) >= 2:
                enriched_pathways.append({
                    'pathway': pathway_name,
                    'overlap': list(overlap),
                    'count': len(overlap)
                })
        
        if not enriched_pathways:
            return self._create_check(
                name="Pathway Enrichment",
                status=ValidationStatus.FAILED,
                message="No significant pathway enrichment detected. Results may be random.",
                confidence=ConfidenceLevel.HIGH,
                evidence={'enriched_pathways': [], 'expected': list(self.CANCER_PATHWAYS.keys())}
            )
        
        # Check for strong enrichment
        strong_enrichment = [p for p in enriched_pathways if p['count'] >= 3]
        
        if strong_enrichment:
            return self._create_check(
                name="Pathway Enrichment",
                status=ValidationStatus.PASSED,
                message=f"Found {len(enriched_pathways)} enriched pathways, {len(strong_enrichment)} with strong overlap",
                confidence=ConfidenceLevel.HIGH,
                evidence={
                    'enriched_pathways': enriched_pathways,
                    'strong_enrichment_count': len(strong_enrichment)
                }
            )
        else:
            return self._create_check(
                name="Pathway Enrichment",
                status=ValidationStatus.WARNING,
                message=f"Found {len(enriched_pathways)} enriched pathways but weak overlap",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'enriched_pathways': enriched_pathways}
            )
    
    def _validate_disease_association(
        self, 
        biomarkers: List[Dict], 
        cancer_type: str
    ) -> ValidationCheck:
        """Validate against known disease biomarkers"""
        known_biomarkers = set()
        
        # Get cancer-specific biomarkers
        cancer_key = cancer_type.lower().replace(' ', '_')
        known_biomarkers.update(self.DISEASE_BIOMARKERS.get(cancer_key, []))
        
        # Also add pan-cancer biomarkers
        for genes in self.DISEASE_BIOMARKERS.values():
            known_biomarkers.update(genes)
        
        biomarker_genes = set([b.get('gene', '').upper() for b in biomarkers])
        matched = biomarker_genes & known_biomarkers
        
        if not matched:
            return self._create_check(
                name="Disease Association",
                status=ValidationStatus.WARNING,
                message="No known disease biomarkers found. These may be novel biomarkers.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={
                    'matched_biomarkers': [],
                    'total_biomarkers': len(biomarker_genes)
                }
            )
        
        # Get importance scores for matched biomarkers
        matched_with_importance = []
        for b in biomarkers:
            if b.get('gene', '').upper() in matched:
                matched_with_importance.append({
                    'gene': b.get('gene'),
                    'importance': b.get('importance', 0)
                })
        
        # Sort by importance
        matched_with_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return self._create_check(
            name="Disease Association",
            status=ValidationStatus.PASSED,
            message=f"Found {len(matched)} known disease biomarkers in top discoveries",
            confidence=ConfidenceLevel.HIGH,
            evidence={
                'matched_biomarkers': matched_with_importance,
                'total_matched': len(matched),
                'known_biomarker_count': len(known_biomarkers)
            }
        )
    
    def _validate_statistical_significance(self, biomarkers: List[Dict]) -> ValidationCheck:
        """Validate statistical significance of biomarkers"""
        # Check if p-values are provided
        has_pvalues = any('p_value' in b or 'pvalue' in b for b in biomarkers)
        
        if not has_pvalues:
            return self._create_check(
                name="Statistical Significance",
                status=ValidationStatus.WARNING,
                message="No p-values provided. Cannot verify statistical significance.",
                confidence=ConfidenceLevel.NONE,
                evidence={'has_pvalues': False}
            )
        
        # Check p-value distribution
        pvalues = []
        for b in biomarkers:
            p = b.get('p_value') or b.get('pvalue') or b.get('p-value')
            if p is not None:
                try:
                    pvalues.append(float(p))
                except (ValueError, TypeError):
                    pass
        
        if not pvalues:
            return self._create_check(
                name="Statistical Significance",
                status=ValidationStatus.WARNING,
                message="Could not parse p-values from biomarkers",
                confidence=ConfidenceLevel.NONE,
                evidence={'pvalue_count': 0}
            )
        
        # Check significance thresholds
        significant = sum(1 for p in pvalues if p < 0.05)
        highly_significant = sum(1 for p in pvalues if p < 0.01)
        
        if significant / len(pvalues) < 0.5:
            return self._create_check(
                name="Statistical Significance",
                status=ValidationStatus.WARNING,
                message=f"Only {significant}/{len(pvalues)} biomarkers meet p<0.05 threshold",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={
                    'significant_count': significant,
                    'total_count': len(pvalues),
                    'min_pvalue': min(pvalues) if pvalues else None
                }
            )
        
        return self._create_check(
            name="Statistical Significance",
            status=ValidationStatus.PASSED,
            message=f"{highly_significant} highly significant (p<0.01), {significant} significant (p<0.05)",
            confidence=ConfidenceLevel.HIGH,
            evidence={
                'significant_count': significant,
                'highly_significant_count': highly_significant,
                'total_count': len(pvalues),
                'min_pvalue': min(pvalues) if pvalues else None
            }
        )
    
    def _validate_pathway_diversity(self, biomarkers: List[Dict]) -> ValidationCheck:
        """Validate diversity of pathways represented"""
        biomarker_genes = [b.get('gene', '').upper() for b in biomarkers]
        
        pathways_covered = []
        for pathway_name, pathway_genes in self.CANCER_PATHWAYS.items():
            overlap = set(biomarker_genes) & set(pathway_genes)
            if overlap:
                pathways_covered.append({
                    'pathway': pathway_name,
                    'genes': list(overlap)
                })
        
        diversity_score = len(pathways_covered) / len(self.CANCER_PATHWAYS)
        
        if diversity_score < 0.1:
            return self._create_check(
                name="Pathway Diversity",
                status=ValidationStatus.WARNING,
                message=f"Low pathway diversity ({len(pathways_covered)} pathways). Results may be biased.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={
                    'pathways_covered': pathways_covered,
                    'diversity_score': diversity_score
                }
            )
        
        return self._create_check(
            name="Pathway Diversity",
            status=ValidationStatus.PASSED,
            message=f"Good pathway diversity with {len(pathways_covered)} cancer-related pathways represented",
            confidence=ConfidenceLevel.HIGH,
            evidence={
                'pathways_covered': pathways_covered,
                'diversity_score': diversity_score
            }
        )
    
    def _validate_with_llm(self, biomarkers: List[Dict], cancer_type: str) -> ValidationCheck:
        """Use LLM for deep validation of biomarkers"""
        try:
            # Get top biomarkers
            top_biomarkers = sorted(
                biomarkers, 
                key=lambda x: x.get('importance', 0), 
                reverse=True
            )[:10]
            
            # Create summary
            biomarker_summary = []
            for b in top_biomarkers:
                gene = b.get('gene', 'Unknown')
                importance = b.get('importance', 0)
                p_value = b.get('p_value', b.get('pvalue', 'N/A'))
                biomarker_summary.append(f"{gene}: importance={importance:.4f}, p={p_value}")
            
            prompt = f"""Analyze these discovered biomarkers for {cancer_type}:

{chr(10).join(biomarker_summary)}

For each biomarker, assess:
1. Is it a known or novel biomarker?
2. Is it druggable (has drug targets)?
3. Is it biologically plausible for this cancer type?

Provide a JSON response:
{{
    "overall_assessment": "brief summary",
    "novel_biomarkers": ["list of potentially novel biomarkers"],
    "druggable_biomarkers": ["list of druggable targets"],
    "biologically_plausible": ["list of plausible biomarkers"],
    "concerns": ["any concerns"],
    "confidence": "high/medium/low"
}}"""

            response = self._query_llm_structured(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "overall_assessment": {"type": "string"},
                        "novel_biomarkers": {"type": "array", "items": {"type": "string"}},
                        "druggable_biomarkers": {"type": "array", "items": {"type": "string"}},
                        "biologically_plausible": {"type": "array", "items": {"type": "string"}},
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
            
            status = ValidationStatus.PASSED if response.get('confidence') != 'low' else ValidationStatus.WARNING
            
            return self._create_check(
                name="LLM Deep Validation",
                status=status,
                message=response.get('overall_assessment', 'LLM validation completed'),
                confidence=conf_map.get(response.get('confidence', 'medium'), ConfidenceLevel.MEDIUM),
                evidence={
                    'novel_biomarkers': response.get('novel_biomarkers', []),
                    'druggable_biomarkers': response.get('druggable_biomarkers', []),
                    'biologically_plausible': response.get('biologically_plausible', []),
                    'concerns': response.get('concerns', [])
                }
            )
            
        except Exception as e:
            logger.warning(f"LLM biomarker validation failed: {e}")
            return self._create_check(
                name="LLM Deep Validation",
                status=ValidationStatus.SKIPPED,
                message=f"LLM validation skipped: {str(e)}",
                confidence=ConfidenceLevel.NONE
            )
    
    def _determine_overall_status(self, checks: List[ValidationCheck]) -> ValidationStatus:
        """Determine overall validation status"""
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
        medium_count = sum(1 for c in checks if c.confidence == ConfidenceLevel.MEDIUM)
        low_count = sum(1 for c in checks if c.confidence == ConfidenceLevel.LOW)
        
        total = len(checks)
        
        if high_count / total >= 0.6:
            return ConfidenceLevel.HIGH
        elif low_count / total >= 0.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.MEDIUM
    
    def _generate_recommendations(
        self, 
        checks: List[ValidationCheck], 
        cancer_type: str,
        biomarkers: List[Dict]
    ) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        for check in checks:
            if check.status == ValidationStatus.FAILED:
                if 'Pathway' in check.name:
                    recommendations.append(
                        "Consider using pathway-aware feature selection"
                    )
                elif 'Disease' in check.name:
                    recommendations.append(
                        "Investigate novel biomarkers through functional studies"
                    )
            elif check.status == ValidationStatus.WARNING:
                if 'Statistical' in check.name:
                    recommendations.append(
                        "Increase sample size or adjust significance threshold"
                    )
                elif 'Diversity' in check.name:
                    recommendations.append(
                        "Consider broader pathway analysis"
                    )
        
        # Add drug-related recommendations
        biomarker_genes = [b.get('gene', '').upper() for b in biomarkers]
        for pathway_name, pathway_genes in self.CANCER_PATHWAYS.items():
            if pathway_name in ['PI3K_AKT', 'MAPK', 'EGFR']:
                overlap = set(biomarker_genes) & set(pathway_genes)
                if overlap:
                    recommendations.append(
                        f"Found genes in {pathway_name} pathway with potential drug targets"
                    )
                    break
        
        return recommendations
    
    def _generate_summary(self, checks: List[ValidationCheck], status: ValidationStatus) -> str:
        """Generate summary"""
        passed = sum(1 for c in checks if c.status == ValidationStatus.PASSED)
        failed = sum(1 for c in checks if c.status == ValidationStatus.FAILED)
        warnings = sum(1 for c in checks if c.status == ValidationStatus.WARNING)
        
        return f"Biomarker validation {status.value}: {passed} passed, {warnings} warnings, {failed} failed"


# Singleton instance
_biomarker_validator = None

def get_biomarker_validator() -> BiomarkerValidator:
    """Get singleton BiomarkerValidator instance"""
    global _biomarker_validator
    if _biomarker_validator is None:
        _biomarker_validator = BiomarkerValidator()
    return _biomarker_validator

