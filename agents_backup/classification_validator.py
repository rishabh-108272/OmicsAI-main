"""
Classification Validation Agent
Validates ML classification results for cancer subtyping
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


class ClassificationValidator(BaseAgent):
    """
    Agent that validates cancer classification results
    
    Performs:
    1. Cross-validation with known cancer biomarkers
    2. Confidence score analysis
    3. Gene expression pattern validation
    4. Medical literature verification
    """
    
    # Known cancer biomarkers for validation
    KNOWN_BIOMARKERS = {
        'colorectal_cancer': {
            'KRAS': 'Proto-oncogene involved in CRC development',
            'APC': 'Tumor suppressor, early CRC marker',
            'TP53': 'Common mutation in advanced CRC',
            'BRAF': 'BRAF V600E mutation in CRC',
            'MSI': 'Microsatellite instability marker',
            'MLH1': 'DNA mismatch repair',
            'MSH2': 'DNA mismatch repair',
            'MSH6': 'DNA mismatch repair',
            'PMS2': 'DNA mismatch repair',
            'EGFR': 'Growth factor receptor',
            'KRAS': 'RAS/MAPK pathway',
            'NRAS': 'RAS/MAPK pathway',
        },
        'liver_cancer': {
            'AFP': 'Alpha-fetoprotein, HCC marker',
            'HBV': 'Hepatitis B virus integration',
            'HCV': 'Hepatitis C virus',
            'TP53': 'Common mutation in HCC',
            'CTNNB1': 'Wnt signaling pathway',
            'AXIN1': 'Wnt signaling pathway',
            'AXIN2': 'Wnt signaling pathway',
            'ARID1A': 'Chromatin remodeling',
            'ARID2': 'Chromatin remodeling',
            'TERT': 'Telomerase activation',
        },
        'lung_cancer': {
            'EGFR': 'Common driver mutation in LUAD',
            'ALK': 'EML4-ALK fusion in LUAD',
            'ROS1': 'ROS1 fusion in LUAD',
            'BRAF': 'BRAF mutations in LUAD',
            'KRAS': 'KRAS mutations in LUAD',
            'TP53': 'Common in both LUAD and LUSC',
            'RB1': 'Loss in LUSC',
            'PTEN': 'Tumor suppressor',
            'NKX2-1': 'LUAD marker',
            'TP63': 'LUSC marker',
        }
    }
    
    def __init__(self):
        super().__init__(
            name="Classification Validator",
            description="Validates cancer classification results against medical knowledge"
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are a medical AI expert specializing in cancer diagnostics and molecular pathology.

Your role is to VALIDATE cancer classification results from ML models. You are NOT a chatbot - you are a reasoning module that:

1. Analyzes gene expression data and classification predictions
2. Cross-references with known cancer biomarkers and pathways
3. Validates the biological plausibility of predictions
4. Identifies potential errors or anomalies
5. Provides evidence-based validation assessments

For each validation check, you must:
- Cite specific evidence (genes, pathways, literature)
- Assign a confidence level (high/medium/low)
- Provide actionable recommendations

Focus on:
- Are the predicted subtypes consistent with the gene expression patterns?
- Are the confidence scores biologically plausible?
- Are there known biomarkers that support or contradict the prediction?
- Is there evidence of sample quality issues?"""
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate classification results
        
        Expected data format:
        {
            'model_type': 'colorectal_cancer' | 'liver_cancer' | 'lung_cancer',
            'predicted_class': str,
            'confidence': float,
            'gene_expression': dict or list,
            'top_genes': list,  # Top discriminating genes
            'patient_id': str,
            'model_performance': dict  # Optional, model metrics
        }
        """
        import time
        start_time = time.time()
        
        try:
            # Extract data
            model_type = data.get('model_type', 'unknown')
            predicted_class = data.get('predicted_class', '')
            confidence = data.get('confidence', 0.0)
            gene_expression = data.get('gene_expression', {})
            top_genes = data.get('top_genes', [])
            patient_id = data.get('patient_id', 'unknown')
            model_performance = data.get('model_performance', {})
            
            checks = []
            recommendations = []
            
            # 1. Check confidence score plausibility
            checks.append(self._validate_confidence(confidence))
            
            # 2. Check if predicted class is valid
            checks.append(self._validate_predicted_class(predicted_class, model_type))
            
            # 3. Validate against known biomarkers
            biomarker_check = self._validate_biomarkers(
                predicted_class, model_type, top_genes, gene_expression
            )
            checks.append(biomarker_check)
            
            # 4. Check gene expression patterns
            expression_check = self._validate_gene_expression(
                top_genes, gene_expression, model_type
            )
            checks.append(expression_check)
            
            # 5. Check model performance (if available)
            if model_performance:
                perf_check = self._validate_model_performance(model_performance, model_type)
                checks.append(perf_check)
            
            # 6. Use LLM for deep validation
            llm_check = self._validate_with_llm(data)
            checks.append(llm_check)
            
            # Determine overall status
            overall_status = self._determine_overall_status(checks)
            
            # Generate recommendations based on checks
            recommendations = self._generate_recommendations(checks, model_type)
            
            processing_time = time.time() - start_time
            
            return ValidationResult(
                agent_name=self.name,
                overall_status=overall_status,
                overall_confidence=self._calculate_overall_confidence(checks),
                summary=self._generate_summary(checks, overall_status),
                checks=checks,
                recommendations=recommendations,
                metadata={
                    'model_type': model_type,
                    'predicted_class': predicted_class,
                    'patient_id': patient_id,
                    'top_genes_analyzed': len(top_genes),
                    'validation_timestamp': datetime.now().isoformat()
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Classification validation error: {e}")
            processing_time = time.time() - start_time
            
            return self._create_result(
                status=ValidationStatus.ERROR,
                summary=f"Validation error: {str(e)}",
                processing_time=processing_time,
                error=str(e)
            )
    
    def _validate_confidence(self, confidence: float) -> ValidationCheck:
        """Validate if confidence score is biologically plausible"""
        if confidence < 0.5:
            return self._create_check(
                name="Confidence Score",
                status=ValidationStatus.WARNING,
                message=f"Low confidence score ({confidence:.2%}). Consider retesting.",
                confidence=ConfidenceLevel.HIGH,
                evidence={'confidence': confidence}
            )
        elif confidence > 0.99:
            return self._create_check(
                name="Confidence Score",
                status=ValidationStatus.WARNING,
                message=f"Very high confidence ({confidence:.2%}). Verify model is not overfitting.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'confidence': confidence}
            )
        else:
            return self._create_check(
                name="Confidence Score",
                status=ValidationStatus.PASSED,
                message=f"Confidence score is within expected range ({confidence:.2%})",
                confidence=ConfidenceLevel.HIGH,
                evidence={'confidence': confidence}
            )
    
    def _validate_predicted_class(self, predicted_class: str, model_type: str) -> ValidationCheck:
        """Validate if predicted class is valid for the model type"""
        valid_classes = {
            'colorectal_cancer': ['CRC', 'Normal', 'Adenocarcinoma', 'Carcinoma', 'High-grade dysplasia'],
            'liver_cancer': ['HCC', 'Normal', 'Cirrhosis', 'CHC', 'Liver Disease'],
            'lung_cancer': ['LUAD', 'LUSC', 'Normal', 'Small Cell', 'Adenocarcinoma', 'Squamous']
        }
        
        valid = valid_classes.get(model_type, [])
        
        # Normalize for comparison
        pred_normalized = predicted_class.strip().upper()
        valid_normalized = [v.upper() for v in valid]
        
        if any(pred_normalized in v or v in pred_normalized for v in valid_normalized):
            return self._create_check(
                name="Predicted Class Validity",
                status=ValidationStatus.PASSED,
                message=f"Predicted class '{predicted_class}' is valid for {model_type}",
                confidence=ConfidenceLevel.HIGH,
                evidence={'predicted_class': predicted_class, 'model_type': model_type}
            )
        else:
            return self._create_check(
                name="Predicted Class Validity",
                status=ValidationStatus.WARNING,
                message=f"Predicted class '{predicted_class}' may not be standard for {model_type}",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'predicted_class': predicted_class, 'model_type': model_type}
            )
    
    def _validate_biomarkers(
        self, 
        predicted_class: str, 
        model_type: str, 
        top_genes: List[str],
        gene_expression: Dict
    ) -> ValidationCheck:
        """Validate against known cancer biomarkers"""
        known_markers = self.KNOWN_BIOMARKERS.get(model_type, {})
        
        # Find matching biomarkers in top genes
        matched_markers = []
        for gene in top_genes:
            if gene.upper() in [k.upper() for k in known_markers.keys()]:
                matched_markers.append(gene)
        
        # Check expression direction for known markers
        supporting_markers = []
        contradicting_markers = []
        
        for gene in matched_markers:
            expr = gene_expression.get(gene, 0)
            # For cancer genes, high expression often indicates oncogenic activity
            # This is simplified - real validation would be more nuanced
            if abs(expr) > 1.5:  # Significant expression
                supporting_markers.append(gene)
            elif abs(expr) < 0.3:
                contradicting_markers.append(gene)
        
        if len(supporting_markers) >= 2:
            return self._create_check(
                name="Biomarker Validation",
                status=ValidationStatus.PASSED,
                message=f"Found {len(supporting_markers)} supporting known biomarkers: {', '.join(supporting_markers[:5])}",
                confidence=ConfidenceLevel.HIGH,
                evidence={
                    'matched_markers': matched_markers,
                    'supporting': supporting_markers,
                    'contradicting': contradicting_markers
                }
            )
        elif len(supporting_markers) == 1:
            return self._create_check(
                name="Biomarker Validation",
                status=ValidationStatus.WARNING,
                message=f"Only 1 supporting biomarker found. Need more evidence.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={
                    'matched_markers': matched_markers,
                    'supporting': supporting_markers
                }
            )
        else:
            return self._create_check(
                name="Biomarker Validation",
                status=ValidationStatus.FAILED,
                message="No known cancer biomarkers found in top genes. Results may be unreliable.",
                confidence=ConfidenceLevel.HIGH,
                evidence={
                    'matched_markers': matched_markers,
                    'expected_markers': list(known_markers.keys())[:10]
                }
            )
    
    def _validate_gene_expression(
        self, 
        top_genes: List[str], 
        gene_expression: Dict,
        model_type: str
    ) -> ValidationCheck:
        """Validate gene expression patterns"""
        if not top_genes:
            return self._create_check(
                name="Gene Expression Pattern",
                status=ValidationStatus.FAILED,
                message="No top genes provided for validation",
                confidence=ConfidenceLevel.NONE
            )
        
        # Check for sufficient genes
        if len(top_genes) < 10:
            return self._create_check(
                name="Gene Expression Pattern",
                status=ValidationStatus.WARNING,
                message=f"Only {len(top_genes)} top genes provided. Minimum 10 recommended.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'gene_count': len(top_genes)}
            )
        
        # Check expression distribution
        expressions = [gene_expression.get(g, 0) for g in top_genes[:20]]
        non_zero = [e for e in expressions if e != 0]
        
        if len(non_zero) / len(expressions) < 0.5:
            return self._create_check(
                name="Gene Expression Pattern",
                status=ValidationStatus.WARNING,
                message="Low proportion of expressed genes. May indicate poor sample quality.",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={
                    'total_genes': len(expressions),
                    'expressed_genes': len(non_zero)
                }
            )
        
        return self._create_check(
            name="Gene Expression Pattern",
            status=ValidationStatus.PASSED,
            message=f"Gene expression patterns appear valid with {len(non_zero)}/{len(expressions)} expressed genes",
            confidence=ConfidenceLevel.MEDIUM,
            evidence={
                'total_genes': len(expressions),
                'expressed_genes': len(non_zero),
                'mean_expression': sum(expressions) / len(expressions) if expressions else 0
            }
        )
    
    def _validate_model_performance(
        self, 
        model_performance: Dict, 
        model_type: str
    ) -> ValidationCheck:
        """Validate model performance metrics"""
        if not model_performance:
            return self._create_check(
                name="Model Performance",
                status=ValidationStatus.SKIPPED,
                message="No model performance metrics available",
                confidence=ConfidenceLevel.NONE
            )
        
        # Check for key metrics
        accuracy = model_performance.get('accuracy') or model_performance.get('Accuracy')
        f1_score = model_performance.get('f1_score') or model_performance.get('F1')
        auc = model_performance.get('auc') or model_performance.get('AUC')
        
        issues = []
        
        if accuracy and accuracy < 0.80:
            issues.append(f"Low accuracy ({accuracy:.2%})")
        if f1_score and f1_score < 0.75:
            issues.append(f"Low F1 score ({f1_score:.2%})")
        if auc and auc < 0.80:
            issues.append(f"Low AUC ({auc:.2%})")
        
        if issues:
            return self._create_check(
                name="Model Performance",
                status=ValidationStatus.WARNING,
                message=f"Model performance concerns: {'; '.join(issues)}",
                confidence=ConfidenceLevel.MEDIUM,
                evidence=model_performance
            )
        
        return self._create_check(
            name="Model Performance",
            status=ValidationStatus.PASSED,
            message="Model performance metrics are within acceptable range",
            confidence=ConfidenceLevel.HIGH,
            evidence=model_performance
        )
    
    def _validate_with_llm(self, data: Dict[str, Any]) -> ValidationCheck:
        """Use LLM for deep validation"""
        try:
            # Prepare data summary
            model_type = data.get('model_type', 'unknown')
            predicted_class = data.get('predicted_class', '')
            confidence = data.get('confidence', 0.0)
            top_genes = data.get('top_genes', [])[:15]
            gene_expr = data.get('gene_expression', {})
            
            # Create summary of top genes with expression
            gene_summary = []
            for gene in top_genes:
                expr = gene_expr.get(gene, 0)
                gene_summary.append(f"{gene}: {expr:.3f}")
            
            prompt = f"""Analyze this cancer classification result and provide validation:

Model Type: {model_type}
Predicted Class: {predicted_class}
Confidence: {confidence:.2%}

Top Genes with Expression Values:
{chr(10).join(gene_summary)}

For each aspect, provide a brief assessment:
1. Biological plausibility
2. Consistency with known cancer pathways
3. Potential concerns

Respond in JSON format:
{{
    "assessment": "brief overall assessment",
    "concerns": ["list of concerns if any"],
    "confidence": "high/medium/low"
}}"""

            response = self._query_llm_structured(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "assessment": {"type": "string"},
                        "concerns": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
                    },
                    "required": ["assessment", "concerns", "confidence"]
                }
            )
            
            # Map LLM confidence
            conf_map = {
                'high': ConfidenceLevel.HIGH,
                'medium': ConfidenceLevel.MEDIUM,
                'low': ConfidenceLevel.LOW
            }
            
            status = ValidationStatus.PASSED if response.get('confidence') != 'low' else ValidationStatus.WARNING
            
            return self._create_check(
                name="LLM Deep Validation",
                status=status,
                message=response.get('assessment', 'LLM validation completed'),
                confidence=conf_map.get(response.get('confidence', 'medium'), ConfidenceLevel.MEDIUM),
                evidence={
                    'llm_assessment': response.get('assessment'),
                    'llm_concerns': response.get('concerns', [])
                }
            )
            
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            return self._create_check(
                name="LLM Deep Validation",
                status=ValidationStatus.SKIPPED,
                message=f"LLM validation skipped: {str(e)}",
                confidence=ConfidenceLevel.NONE
            )
    
    def _determine_overall_status(self, checks: List[ValidationCheck]) -> ValidationStatus:
        """Determine overall validation status from checks"""
        if any(c.status == ValidationStatus.ERROR for c in checks):
            return ValidationStatus.ERROR
        if any(c.status == ValidationStatus.FAILED for c in checks):
            return ValidationStatus.FAILED
        if any(c.status == ValidationStatus.WARNING for c in checks):
            return ValidationStatus.WARNING
        return ValidationStatus.PASSED
    
    def _calculate_overall_confidence(self, checks: List[ValidationCheck]) -> ConfidenceLevel:
        """Calculate overall confidence from checks"""
        if not checks:
            return ConfidenceLevel.NONE
        
        high_count = sum(1 for c in checks if c.confidence == ConfidenceLevel.HIGH)
        medium_count = sum(1 for c in checks if c.confidence == ConfidenceLevel.MEDIUM)
        low_count = sum(1 for c in checks if c.confidence == ConfidenceLevel.LOW)
        
        total = len(checks)
        
        if high_count / total >= 0.7:
            return ConfidenceLevel.HIGH
        elif low_count / total >= 0.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.MEDIUM
    
    def _generate_recommendations(self, checks: List[ValidationCheck], model_type: str) -> List[str]:
        """Generate recommendations based on failed/warning checks"""
        recommendations = []
        
        for check in checks:
            if check.status == ValidationStatus.FAILED:
                if 'Biomarker' in check.name:
                    recommendations.append(
                        "Consider running biomarker discovery to identify novel markers"
                    )
                elif 'Gene Expression' in check.name:
                    recommendations.append(
                        "Review sample quality and data preprocessing"
                    )
                elif 'Confidence' in check.name:
                    recommendations.append(
                        "Collect additional samples for validation"
                    )
            elif check.status == ValidationStatus.WARNING:
                if 'Biomarker' in check.name:
                    recommendations.append(
                        "Consider orthogonal validation of biomarkers"
                    )
                elif 'Model Performance' in check.name:
                    recommendations.append(
                        "Consider retraining model with more data"
                    )
        
        return recommendations
    
    def _generate_summary(self, checks: List[ValidationCheck], status: ValidationStatus) -> str:
        """Generate summary message"""
        passed = sum(1 for c in checks if c.status == ValidationStatus.PASSED)
        failed = sum(1 for c in checks if c.status == ValidationStatus.FAILED)
        warnings = sum(1 for c in checks if c.status == ValidationStatus.WARNING)
        
        return f"Validation {status.value}: {passed} passed, {warnings} warnings, {failed} failed"


# Singleton instance
_classification_validator = None

def get_classification_validator() -> ClassificationValidator:
    """Get singleton ClassificationValidator instance"""
    global _classification_validator
    if _classification_validator is None:
        _classification_validator = ClassificationValidator()
    return _classification_validator

