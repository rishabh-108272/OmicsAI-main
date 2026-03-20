"""
Protein Structure Validation Agent
Validates AlphaFold predictions and 3D protein structures
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
    UniProtClient,
    ExternalValidator,
    validate_protein_external
)

logger = logging.getLogger(__name__)


class ProteinValidator(BaseAgent):
    """
    Agent that validates AlphaFold protein structure predictions
    
    Performs:
    1. pLDDT confidence score validation
    2. Structural quality assessment
    3. Domain organization verification
    4. Known structure comparison
    5. Disorder region identification
    """
    
    # Typical pLDDT thresholds
    PLDDT_THRESHOLDS = {
        'very_high': 90,    # Reliable for drug binding
        'high': 70,         # Reliable for backbone
        'low': 50,          # Unreliable regions
        'very_low': 0       # Disorder
    }
    
    # Known protein domains/motifs for validation
    KNOWN_DOMAINS = {
        'EGFR': {
            'domains': ['Receptor tyrosine kinase', 'Tyrosine kinase domain'],
            'length_range': (1210, 1230),
            'structure': 'Transmembrane'
        },
        'KRAS': {
            'domains': ['Small GTPase', 'P-loop NTP hydrolase'],
            'length_range': (188, 210),
            'structure': 'G-domain'
        },
        'TP53': {
            'domains': ['p53 DNA-binding domain', 'Transactivation domain'],
            'length_range': (390, 420),
            'structure': 'Tetramer'
        },
        'BRCA1': {
            'domains': ['RING domain', 'BRCT domain'],
            'length_range': (1860, 1900),
            'structure': 'Multi-domain'
        },
        'ALK': {
            'domains': ['Receptor tyrosine kinase', 'Tyrosine kinase domain'],
            'length_range': (1620, 1760),
            'structure': 'Transmembrane'
        },
        'BRAF': {
            'domains': ['Protein kinase', 'RBD', 'CRD'],
            'length_range': (765, 800),
            'structure': 'Kinase domain'
        },
        'PIK3CA': {
            'domains': ['PI3K/PI4K', 'Kinase domain'],
            'length_range': (1040, 1080),
            'structure': 'Multi-domain'
        },
        'MTOR': {
            'domains': ['PI3K-related kinase', 'FRB domain', 'Kinase domain'],
            'length_range': (2540, 2580),
            'structure': 'Multi-domain'
        },
    }
    
    # Quality indicators
    QUALITY_INDICATORS = {
        'excellent': {'min_plddt': 90, 'min_coverage': 0.95},
        'good': {'min_plddt': 70, 'min_coverage': 0.85},
        'moderate': {'min_plddt': 50, 'min_coverage': 0.70},
        'poor': {'min_plddt': 0, 'min_coverage': 0.50}
    }
    
    def __init__(self):
        super().__init__(
            name="Protein Validator",
            description="Validates AlphaFold protein structure predictions"
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are a structural bioinformatics expert specializing in protein structure validation.

Your role is to VALIDATE AlphaFold protein structure predictions. You are NOT a chatbot - you are a reasoning module that:

1. Analyzes pLDDT confidence scores
2. Validates structural quality and domain organization
3. Checks for disordered regions
4. Compares with known structures
5. Identifies potential modeling issues

For each structure, you must:
- Assess overall quality based on pLDDT distribution
- Identify reliable vs unreliable regions
- Check for domain boundaries
- Flag potential modeling errors

Focus on:
- Is the pLDDT distribution consistent with the protein type?
- Are the disordered regions biologically reasonable?
- Are there any red flags (e.g., low confidence in known functional regions)?
- Is the structure suitable for drug discovery applications?"""
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate protein structure prediction
        
        Expected data format:
        {
            'protein_id': str,  # UniProt accession
            'protein_name': str,
            'sequence': str,
            'plddt_scores': list,  # Per-residue pLDDT
            'pae_scores': list,    # Per-residue PAE (optional)
            'structure_data': dict  # Optional structure metrics
        }
        """
        import time
        start_time = time.time()
        
        try:
            # Extract data
            protein_id = data.get('protein_id', 'unknown')
            protein_name = data.get('protein_name', '')
            sequence = data.get('sequence', '')
            plddt_scores = data.get('plddt_scores', [])
            pae_scores = data.get('pae_scores', [])
            structure_data = data.get('structure_data', {})
            
            checks = []
            recommendations = []
            
            if not plddt_scores:
                checks.append(self._create_check(
                    name="pLDDT Scores",
                    status=ValidationStatus.FAILED,
                    message="No pLDDT scores provided for validation",
                    confidence=ConfidenceLevel.NONE
                ))
                
                return self._create_result(
                    status=ValidationStatus.FAILED,
                    summary="No pLDDT scores to validate",
                    checks=checks,
                    processing_time=time.time() - start_time
                )
            
            # 1. Validate overall quality
            quality_check = self._validate_overall_quality(plddt_scores)
            checks.append(quality_check)
            
            # 2. Validate pLDDT distribution
            dist_check = self._validate_plddt_distribution(plddt_scores)
            checks.append(dist_check)
            
            # 3. Validate sequence length
            length_check = self._validate_sequence_length(sequence, protein_name)
            checks.append(length_check)
            
            # 4. Validate disordered regions
            disorder_check = self._validate_disordered_regions(plddt_scores)
            checks.append(disorder_check)
            
            # 5. Validate PAE if available
            if pae_scores:
                pae_check = self._validate_pae_scores(pae_scores)
                checks.append(pae_check)
            
    # 6. Validate against known domains
            domain_check = self._validate_known_domains(protein_name, plddt_scores, sequence)
            checks.append(domain_check)
            
            # 7. External API validation (UniProt, KEGG, etc.)
            external_check = self._validate_with_external_apis(protein_id, protein_name, plddt_scores, sequence)
            checks.append(external_check)
            
            # 8. LLM deep validation
            llm_check = self._validate_with_llm(
                protein_id, protein_name, plddt_scores, sequence
            )
            checks.append(llm_check)
            
            # Determine overall status
            overall_status = self._determine_overall_status(checks)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                checks, plddt_scores, protein_name
            )
            
            processing_time = time.time() - start_time
            
            return ValidationResult(
                agent_name=self.name,
                overall_status=overall_status,
                overall_confidence=self._calculate_overall_confidence(checks),
                summary=self._generate_summary(checks, overall_status),
                checks=checks,
                recommendations=recommendations,
                metadata={
                    'protein_id': protein_id,
                    'protein_name': protein_name,
                    'sequence_length': len(sequence),
                    'validation_timestamp': datetime.now().isoformat()
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Protein validation error: {e}")
            processing_time = time.time() - start_time
            
            return self._create_result(
                status=ValidationStatus.ERROR,
                summary=f"Validation error: {str(e)}",
                processing_time=processing_time,
                error=str(e)
            )
    
    def _validate_overall_quality(self, plddt_scores: List[float]) -> ValidationCheck:
        """Validate overall prediction quality"""
        if not plddt_scores:
            return self._create_check(
                name="Overall Quality",
                status=ValidationStatus.FAILED,
                message="No pLDDT scores to analyze",
                confidence=ConfidenceLevel.NONE
            )
        
        # Calculate metrics
        mean_plddt = sum(plddt_scores) / len(plddt_scores)
        median_plddt = sorted(plddt_scores)[len(plddt_scores) // 2]
        
        # Count regions by confidence
        very_high = sum(1 for p in plddt_scores if p >= 90)
        high = sum(1 for p in plddt_scores if 70 <= p < 90)
        low = sum(1 for p in plddt_scores if 50 <= p < 50)
        very_low = sum(1 for p in plddt_scores if p < 50)
        
        total = len(plddt_scores)
        
        # Determine quality category
        if mean_plddt >= 90 and very_high / total >= 0.70:
            quality = 'excellent'
            status = ValidationStatus.PASSED
            confidence = ConfidenceLevel.HIGH
        elif mean_plddt >= 70 and (very_high + high) / total >= 0.70:
            quality = 'good'
            status = ValidationStatus.PASSED
            confidence = ConfidenceLevel.HIGH
        elif mean_plddt >= 50 and (very_high + high + low) / total >= 0.50:
            quality = 'moderate'
            status = ValidationStatus.WARNING
            confidence = ConfidenceLevel.MEDIUM
        else:
            quality = 'poor'
            status = ValidationStatus.FAILED
            confidence = ConfidenceLevel.LOW
        
        return self._create_check(
            name="Overall Quality",
            status=status,
            message=f"Structure quality: {quality} (mean pLDDT: {mean_plddt:.1f})",
            confidence=confidence,
            evidence={
                'quality': quality,
                'mean_plddt': mean_plddt,
                'median_plddt': median_plddt,
                'very_high_count': very_high,
                'high_count': high,
                'low_count': low,
                'very_low_count': very_low,
                'coverage': (very_high + high) / total if total > 0 else 0
            }
        )
    
    def _validate_plddt_distribution(self, plddt_scores: List[float]) -> ValidationCheck:
        """Validate pLDDT score distribution"""
        if not plddt_scores:
            return self._create_check(
                name="pLDDT Distribution",
                status=ValidationStatus.FAILED,
                message="No pLDDT scores to analyze",
                confidence=ConfidenceLevel.NONE
            )
        
        # Check for problematic patterns
        # 1. All scores very low - indicates potential issues
        if all(p < 50 for p in plddt_scores):
            return self._create_check(
                name="pLDDT Distribution",
                status=ValidationStatus.FAILED,
                message="All residues have very low confidence. Structure may be unreliable.",
                confidence=ConfidenceLevel.HIGH,
                evidence={'pattern': 'all_low'}
            )
        
        # 2. Check for bimodal distribution (could indicate domain boundary issues)
        import statistics
        try:
            std_dev = statistics.stdev(plddt_scores)
        except:
            std_dev = 0
        
        # 3. Check for concerning regions at N/C termini
        n_term = sum(plddt_scores[:min(20, len(plddt_scores)//10)]) / min(20, len(plddt_scores)//10) if plddt_scores else 0
        c_term = sum(plddt_scores[-min(20, len(plddt_scores)//10):]) / min(20, len(plddt_scores)//10) if plddt_scores else 0
        
        concerns = []
        if n_term < 30:
            concerns.append("Low confidence N-terminus")
        if c_term < 30:
            concerns.append("Low confidence C-terminus")
        
        if concerns:
            return self._create_check(
                name="pLDDT Distribution",
                status=ValidationStatus.WARNING,
                message=f"Distribution concerns: {'; '.join(concerns)}",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={
                    'n_term_plddt': n_term,
                    'c_term_plddt': c_term,
                    'std_dev': std_dev,
                    'concerns': concerns
                }
            )
        
        return self._create_check(
            name="pLDDT Distribution",
            status=ValidationStatus.PASSED,
            message="pLDDT distribution appears normal",
            confidence=ConfidenceLevel.MEDIUM,
            evidence={
                'n_term_plddt': n_term,
                'c_term_plddt': c_term,
                'std_dev': std_dev
            }
        )
    
    def _validate_sequence_length(
        self, 
        sequence: str, 
        protein_name: str
    ) -> ValidationCheck:
        """Validate sequence length"""
        if not sequence:
            return self._create_check(
                name="Sequence Length",
                status=ValidationStatus.WARNING,
                message="No sequence provided",
                confidence=ConfidenceLevel.NONE
            )
        
        length = len(sequence)
        
        # Check if known protein
        known_info = self.KNOWN_DOMAINS.get(protein_name.upper())
        
        if known_info:
            expected_range = known_info.get('length_range', (0, 10000))
            if expected_range[0] <= length <= expected_range[1]:
                return self._create_check(
                    name="Sequence Length",
                    status=ValidationStatus.PASSED,
                    message=f"Length {length} matches expected range for {protein_name}",
                    confidence=ConfidenceLevel.HIGH,
                    evidence={
                        'length': length,
                        'expected_range': expected_range,
                        'protein_known': True
                    }
                )
            else:
                return self._create_check(
                    name="Sequence Length",
                    status=ValidationStatus.WARNING,
                    message=f"Length {length} differs from expected {expected_range} for {protein_name}",
                    confidence=ConfidenceLevel.MEDIUM,
                    evidence={
                        'length': length,
                        'expected_range': expected_range,
                        'protein_known': True
                    }
                )
        
        # General length checks
        if length < 50:
            return self._create_check(
                name="Sequence Length",
                status=ValidationStatus.WARNING,
                message=f"Very short sequence ({length} aa)",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'length': length}
            )
        elif length > 5000:
            return self._create_check(
                name="Sequence Length",
                status=ValidationStatus.WARNING,
                message=f"Very long sequence ({length} aa). May have modeling challenges.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'length': length}
            )
        
        return self._create_check(
            name="Sequence Length",
            status=ValidationStatus.PASSED,
            message=f"Sequence length {length} aa is within normal range",
            confidence=ConfidenceLevel.MEDIUM,
            evidence={'length': length}
        )
    
    def _validate_disordered_regions(self, plddt_scores: List[float]) -> ValidationCheck:
        """Validate disordered regions"""
        if not plddt_scores:
            return self._create_check(
                name="Disordered Regions",
                status=ValidationStatus.FAILED,
                message="No pLDDT scores to analyze",
                confidence=ConfidenceLevel.NONE
            )
        
        # Identify disordered regions (pLDDT < 50)
        disordered = []
        current_region = None
        
        for i, score in enumerate(plddt_scores):
            if score < 50:
                if current_region is None:
                    current_region = {'start': i, 'end': i, 'length': 1}
                else:
                    current_region['end'] = i
                    current_region['length'] += 1
            else:
                if current_region is not None:
                    disordered.append(current_region)
                    current_region = None
        
        if current_region is not None:
            disordered.append(current_region)
        
        total_disordered = sum(r['length'] for r in disordered)
        disordered_fraction = total_disordered / len(plddt_scores) if plddt_scores else 0
        
        # Large disordered regions are concerning for structured proteins
        large_disorder = [r for r in disordered if r['length'] > 50]
        
        if disordered_fraction > 0.5:
            return self._create_check(
                name="Disordered Regions",
                status=ValidationStatus.WARNING,
                message=f"High disorder: {disordered_fraction:.1%} of structure is disordered",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={
                    'disordered_regions': disordered,
                    'total_disordered': total_disordered,
                    'disordered_fraction': disordered_fraction,
                    'large_disordered_regions': large_disorder
                }
            )
        elif large_disorder:
            return self._create_check(
                name="Disordered Regions",
                status=ValidationStatus.WARNING,
                message=f"Found {len(large_disorder)} large disordered regions (>50 aa)",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={
                    'disordered_regions': disordered,
                    'large_disordered_regions': large_disorder
                }
            )
        
        return self._create_check(
            name="Disordered Regions",
            status=ValidationStatus.PASSED,
            message=f"Disorder content ({disordered_fraction:.1%}) is reasonable",
            confidence=ConfidenceLevel.MEDIUM,
            evidence={
                'disordered_regions': disordered,
                'disordered_fraction': disordered_fraction
            }
        )
    
    def _validate_pae_scores(self, pae_scores: List[List[float]]) -> ValidationCheck:
        """Validate Predicted Aligned Error (PAE) scores"""
        if not pae_scores:
            return self._create_check(
                name="PAE Scores",
                status=ValidationStatus.SKIPPED,
                message="No PAE scores available",
                confidence=ConfidenceLevel.NONE
            )
        
        # PAE is a matrix - get average
        try:
            flat_pae = [p for row in pae_scores for p in row]
            mean_pae = sum(flat_pae) / len(flat_pae) if flat_pae else float('inf')
            
            if mean_pae < 5:
                return self._create_check(
                    name="PAE Scores",
                    status=ValidationStatus.PASSED,
                    message=f"Low PAE ({mean_pae:.1f}) indicates well-confident relative positions",
                    confidence=ConfidenceLevel.HIGH,
                    evidence={'mean_pae': mean_pae}
                )
            elif mean_pae < 15:
                return self._create_check(
                    name="PAE Scores",
                    status=ValidationStatus.PASSED,
                    message=f"Moderate PAE ({mean_pae:.1f}) - some uncertain domain orientations",
                    confidence=ConfidenceLevel.MEDIUM,
                    evidence={'mean_pae': mean_pae}
                )
            else:
                return self._create_check(
                    name="PAE Scores",
                    status=ValidationStatus.WARNING,
                    message=f"High PAE ({mean_pae:.1f}) - domain orientations uncertain",
                    confidence=ConfidenceLevel.MEDIUM,
                    evidence={'mean_pae': mean_pae}
                )
        except Exception as e:
            return self._create_check(
                name="PAE Scores",
                status=ValidationStatus.SKIPPED,
                message=f"Could not analyze PAE: {str(e)}",
                confidence=ConfidenceLevel.NONE
            )
    
    def _validate_known_domains(
        self, 
        protein_name: str, 
        plddt_scores: List[float],
        sequence: str
    ) -> ValidationCheck:
        """Validate against known domain structures"""
        known_info = self.KNOWN_DOMAINS.get(protein_name.upper())
        
        if not known_info:
            return self._create_check(
                name="Known Domains",
                status=ValidationStatus.SKIPPED,
                message=f"No known domain info for {protein_name}",
                confidence=ConfidenceLevel.NONE,
                evidence={'protein_known': False}
            )
        
        # Check if kinase domains have high confidence
        domains = known_info.get('domains', [])
        
        if any('kinase' in d.lower() for d in domains):
            # Kinases typically have well-structured kinase domains
            # Check if middle region (kinase domain) has high confidence
            seq_len = len(plddt_scores)
            middle_start = seq_len // 3
            middle_end = 2 * seq_len // 3
            
            middle_scores = plddt_scores[middle_start:middle_end]
            middle_mean = sum(middle_scores) / len(middle_scores) if middle_scores else 0
            
            if middle_mean >= 70:
                return self._create_check(
                    name="Known Domains",
                    status=ValidationStatus.PASSED,
                    message=f"Kinase domain region has high confidence (mean: {middle_mean:.1f})",
                    confidence=ConfidenceLevel.HIGH,
                    evidence={
                        'domain_type': 'kinase',
                        'middle_region_mean': middle_mean
                    }
                )
            else:
                return self._create_check(
                    name="Known Domains",
                    status=ValidationStatus.WARNING,
                    message=f"Kinase domain region has low confidence (mean: {middle_mean:.1f})",
                    confidence=ConfidenceLevel.MEDIUM,
                    evidence={
                        'domain_type': 'kinase',
                        'middle_region_mean': middle_mean
                    }
                )
        
        return self._create_check(
            name="Known Domains",
            status=ValidationStatus.PASSED,
            message=f"Known domains validated for {protein_name}",
            confidence=ConfidenceLevel.MEDIUM,
            evidence={'domains': domains}
        )
    
    def _validate_with_external_apis(
        self, 
        protein_id: str, 
        protein_name: str,
        plddt_scores: List[float],
        sequence: str
    ) -> ValidationCheck:
        """Validate using external APIs (UniProt, KEGG)"""
        try:
            # Query UniProt for protein information
            uniprot_result = validate_protein_external(protein_id, plddt_scores)
            
            if uniprot_result.get('validated') and uniprot_result.get('protein_info'):
                protein_info = uniprot_result['protein_info']
                
                # Check sequence length match
                sequence_matched = uniprot_result.get('sequence_matched', False)
                warning = uniprot_result.get('warning', '')
                
                if sequence_matched:
                    return self._create_check(
                        name="External Database Validation",
                        status=ValidationStatus.PASSED,
                        message=f"Validated against UniProt: {protein_info.get('protein_name', protein_name)} ({protein_info.get('organism', 'Unknown')})",
                        confidence=ConfidenceLevel.HIGH,
                        evidence={
                            'source': 'uniprot',
                            'protein_info': protein_info,
                            'sequence_matched': True
                        }
                    )
                else:
                    return self._create_check(
                        name="External Database Validation",
                        status=ValidationStatus.WARNING,
                        message=f"UniProt match with warning: {warning}",
                        confidence=ConfidenceLevel.MEDIUM,
                        evidence={
                            'source': 'uniprot',
                            'protein_info': protein_info,
                            'warning': warning
                        }
                    )
            
            # Try searching by protein name if accession didn't work
            if protein_name:
                search_results = UniProtClient.search_proteins(protein_name)
                if search_results and len(search_results) > 0:
                    return self._create_check(
                        name="External Database Validation",
                        status=ValidationStatus.PASSED,
                        message=f"Found {len(search_results)} UniProt matches for {protein_name}",
                        confidence=ConfidenceLevel.MEDIUM,
                        evidence={
                            'source': 'uniprot_search',
                            'match_count': len(search_results)
                        }
                    )
            
            return self._create_check(
                name="External Database Validation",
                status=ValidationStatus.SKIPPED,
                message=f"Could not validate against external databases (UniProt)",
                confidence=ConfidenceLevel.NONE,
                evidence={'reason': 'No external match found'}
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
        protein_id: str, 
        protein_name: str,
        plddt_scores: List[float],
        sequence: str
    ) -> ValidationCheck:
        """Use LLM for deep validation"""
        try:
            # Calculate summary statistics
            mean_plddt = sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0
            very_high = sum(1 for p in plddt_scores if p >= 90)
            high = sum(1 for p in plddt_scores if 70 <= p < 90)
            low = sum(1 for p in plddt_scores if 50 <= p < 70)
            very_low = sum(1 for p in plddt_scores if p < 50)
            
            total = len(plddt_scores)
            
            # Identify problematic regions (very low confidence)
            problem_regions = []
            if very_low / total > 0.3:
                problem_regions.append("High proportion of very low confidence residues")
            if mean_plddt < 50:
                problem_regions.append("Overall low confidence")
            
            prompt = f"""Analyze this AlphaFold protein structure prediction:

Protein: {protein_name} ({protein_id})
Sequence Length: {len(sequence)} aa

pLDDT Statistics:
- Mean: {mean_plddt:.1f}
- Very High (≥90): {very_high} residues ({very_high/total*100:.1f}%)
- High (70-90): {high} residues ({high/total*100:.1f}%)
- Low (50-70): {low} residues ({low/total*100:.1f}%)
- Very Low (<50): {very_low} residues ({very_low/total*100:.1f}%)

Issues: {', '.join(problem_regions) if problem_regions else 'None identified'}

Assess:
1. Overall reliability for drug discovery
2. Whether regions of interest are well-modeled
3. Potential limitations

Provide JSON response:
{{
    "overall_assessment": "brief summary",
    "drug_discovery_suitable": true/false,
    "reliable_regions": ["list of reliable functional regions"],
    "unreliable_regions": ["list of regions to interpret with caution"],
    "limitations": ["list of limitations"],
    "confidence": "high/medium/low"
}}"""

            response = self._query_llm_structured(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "overall_assessment": {"type": "string"},
                        "drug_discovery_suitable": {"type": "boolean"},
                        "reliable_regions": {"type": "array", "items": {"type": "string"}},
                        "unreliable_regions": {"type": "array", "items": {"type": "string"}},
                        "limitations": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
                    }
                }
            )
            
            conf_map = {
                'high': ConfidenceLevel.HIGH,
                'medium': ConfidenceLevel.MEDIUM,
                'low': ConfidenceLevel.LOW
            }
            
            suitable = response.get('drug_discovery_suitable', True)
            status = ValidationStatus.PASSED if suitable else ValidationStatus.WARNING
            
            return self._create_check(
                name="LLM Deep Validation",
                status=status,
                message=response.get('overall_assessment', 'LLM validation completed'),
                confidence=conf_map.get(response.get('confidence', 'medium'), ConfidenceLevel.MEDIUM),
                evidence={
                    'drug_discovery_suitable': suitable,
                    'reliable_regions': response.get('reliable_regions', []),
                    'unreliable_regions': response.get('unreliable_regions', []),
                    'limitations': response.get('limitations', [])
                }
            )
            
        except Exception as e:
            logger.warning(f"LLM protein validation failed: {e}")
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
        plddt_scores: List[float],
        protein_name: str
    ) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        for check in checks:
            if check.status == ValidationStatus.FAILED:
                if 'Quality' in check.name:
                    recommendations.append(
                        "Consider experimental structure determination"
                    )
                elif 'pLDDT' in check.name:
                    recommendations.append(
                        "Results should be interpreted with caution"
                    )
            elif check.status == ValidationStatus.WARNING:
                if 'Disordered' in check.name:
                    recommendations.append(
                        "Disordered regions may be functionally important"
                    )
                elif 'Domains' in check.name:
                    recommendations.append(
                        "Validate domain boundaries experimentally"
                    )
        
        # Add specific recommendations based on quality
        mean_plddt = sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0
        
        if mean_plddt >= 90:
            recommendations.append(
                "Structure is highly reliable for drug discovery applications"
            )
        elif mean_plddt >= 70:
            recommendations.append(
                "Structure is suitable for most computational analyses"
            )
        elif mean_plddt >= 50:
            recommendations.append(
                "Focus on high-confidence regions for downstream analysis"
            )
        else:
            recommendations.append(
                "Structure has limited reliability. Consider experimental validation."
            )
        
        return recommendations
    
    def _generate_summary(self, checks: List[ValidationCheck], status: ValidationStatus) -> str:
        """Generate summary"""
        passed = sum(1 for c in checks if c.status == ValidationStatus.PASSED)
        failed = sum(1 for c in checks if c.status == ValidationStatus.FAILED)
        warnings = sum(1 for c in checks if c.status == ValidationStatus.WARNING)
        
        return f"Protein validation {status.value}: {passed} passed, {warnings} warnings, {failed} failed"


# Singleton instance
_protein_validator = None

def get_protein_validator() -> ProteinValidator:
    """Get singleton ProteinValidator instance"""
    global _protein_validator
    if _protein_validator is None:
        _protein_validator = ProteinValidator()
    return _protein_validator

