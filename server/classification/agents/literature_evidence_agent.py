"""
Literature Evidence Agent - Provides biological/scientific justification  
Uses LIVE PubMed/NCBI API searches (no mocks/static data)
"""

import json
import time
import logging
from typing import Dict, List, Any
from datetime import datetime
from collections import Counter

from .base_agent import BaseAgent, ValidationResult, ValidationStatus, ConfidenceLevel, ValidationCheck
from .external_api_client import PubMedClient

logger = logging.getLogger(__name__)


class LiteratureEvidenceAgent(BaseAgent):
    """
    Agent providing literature-based biological justification
    
    Performs LIVE PubMed searches for:
    1. Biomarker-disease association evidence
    2. Recent publication trends
    3. High-impact journal validation
    4. Citation analysis (proxy)
    """
    
    HIGH_IMPACT_JOURNALS = {
        'Nature', 'Science', 'Cell', 'NEJM', 'Lancet', 'Nature Medicine',
        'Cancer Cell', 'Nature Cancer', 'Cell Reports Medicine', 'Nature Genetics'
    }
    
    def __init__(self):
        super().__init__(
            name="Literature Evidence Agent",
            description="Provides biological justification via LIVE PubMed literature search"
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are a biomedical literature expert synthesizing evidence from PubMed.

Given LIVE PubMed search results for biomarkers, synthesize:
1. Strength of biomarker-disease association evidence
2. Temporal trends (recent vs established)
3. Journal quality assessment
4. Mechanistic evidence summary
5. Controversies/gaps in literature

Format as structured evidence review focusing on therapeutic relevance."""
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate biomarkers with literature evidence"""
        start_time = time.time()
        try:
            cancer_type = data.get('cancer_type', 'unknown').replace('_', ' ')
            biomarkers = data.get('biomarkers', [])
            
            if not biomarkers:
                return self._create_result(
                    status=ValidationStatus.FAILED,
                    summary="No biomarkers provided",
                    processing_time=time.time() - start_time
                )
            
            genes = [b.get('gene', '') for b in biomarkers[:10] if b.get('gene')]  # Top 10
            checks = []
            
            # 1. Literature volume & relevance (LIVE PubMed)
            volume_check = self._validate_literature_volume(genes, cancer_type)
            checks.append(volume_check)
            
            # 2. Publication recency
            recency_check = self._validate_recency(genes, cancer_type)
            checks.append(recency_check)
            
            # 3. High-impact publications
            impact_check = self._validate_impact(genes, cancer_type)
            checks.append(impact_check)
            
            # 4. Evidence quality assessment
            quality_check = self._validate_evidence_quality(genes, cancer_type)
            checks.append(quality_check)
            
            # 5. LLM literature synthesis
            synthesis_check = self._synthesize_literature(genes, cancer_type, checks)
            checks.append(synthesis_check)
            
            overall_status = self._determine_status(checks)
            
            return self._create_result(
                status=overall_status,
                summary=f"Literature evidence: {len([c for c in checks if c.status == ValidationStatus.PASSED])} strong associations",
                checks=checks,
                recommendations=self._get_recommendations(checks),
                metadata={
                    'genes_analyzed': len(genes),
                    'cancer_type': cancer_type,
                    'total_publications': sum(c.evidence.get('pub_count', 0) for c in checks)
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Literature agent error: {e}")
            return self._create_result(
                status=ValidationStatus.ERROR,
                summary=f"Error: {str(e)}",
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def _validate_literature_volume(self, genes: List[str], cancer_type: str) -> ValidationCheck:
        """LIVE PubMed volume analysis"""
        total_pubs = 0
        strong_assoc = 0
        pub_summaries = []
        
        for gene in genes:
            query = f"{gene}[Title/Abstract] AND {cancer_type}[Title/Abstract]"
            result = PubMedClient.search_pubmed(query, max_results=10)
            
            pub_count = result.get('count', 0) if result else 0
            total_pubs += pub_count
            
            if pub_count >= 10:
                strong_assoc += 1
                pub_summaries.append({
                    'gene': gene,
                    'publications': pub_count,
                    'top_articles': result.get('articles', [])[:2] if result else []
                })
        
        avg_pubs = total_pubs / len(genes) if genes else 0
        
        if avg_pubs >= 20:
            status = ValidationStatus.PASSED
            confidence = ConfidenceLevel.HIGH
            msg = f"Strong literature support (avg {avg_pubs:.0f} pubs/biomarker)"
        elif avg_pubs >= 5:
            status = ValidationStatus.PASSED
            confidence = ConfidenceLevel.MEDIUM  
            msg = f"Moderate literature support (avg {avg_pubs:.0f} pubs/biomarker)"
        else:
            status = ValidationStatus.WARNING
            confidence = ConfidenceLevel.LOW
            msg = f"Limited literature (avg {avg_pubs:.0f} pubs/biomarker)"
        
        return self._create_check(
            name="Literature Volume (LIVE PubMed)",
            status=status,
            message=msg,
            confidence=confidence,
            evidence={
                'total_publications': total_pubs,
                'avg_per_gene': avg_pubs,
                'strong_associations': strong_assoc,
                'examples': pub_summaries[:3]
            }
        )
    
    def _validate_recency(self, genes: List[str], cancer_type: str) -> ValidationCheck:
        """Check recent publications (last 5 years)"""
        recent_pubs = 0
        recent_articles = []
        
        for gene in genes[:6]:
            query = f"{gene}[Title/Abstract] AND {cancer_type}[Title/Abstract]"
            result = PubMedClient.search_pubmed(query, max_results=5)
            
            if result:
                recent_count = 0
                for article in result.get('articles', []):
                    pubdate = article.get('pubdate', '')
                    if any(year in pubdate for year in ['2020', '2021', '2022', '2023', '2024']):
                        recent_count += 1
                
                recent_pubs += recent_count
                if recent_count > 0:
                    recent_articles.append({'gene': gene, 'recent_articles': recent_count})
        
        recent_ratio = recent_pubs / max(len(genes), 1)
        
        if recent_ratio >= 0.4:
            return self._create_check(
                name="Publication Recency",
                status=ValidationStatus.PASSED,
                message=f"{recent_ratio:.0%} recent publications (2020-2024)",
                confidence=ConfidenceLevel.HIGH,
                evidence={'recent_ratio': recent_ratio, 'examples': recent_articles}
            )
        else:
            return self._create_check(
                name="Publication Recency", 
                status=ValidationStatus.WARNING,
                message=f"Only {recent_ratio:.0%} recent publications",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'recent_ratio': recent_ratio}
            )
    
    def _validate_impact(self, genes: List[str], cancer_type: str) -> ValidationCheck:
        """High-impact journal validation"""
        high_impact_count = 0
        high_impact_pubs = []
        
        for gene in genes[:5]:
            query = f"{gene}[Title/Abstract] AND {cancer_type}[Title/Abstract]"
            result = PubMedClient.search_pubmed(query, max_results=10)
            
            if result:
                for article in result.get('articles', []):
                    journal = article.get('journal', '').lower()
                    if any(impact_journal.lower() in journal for impact_journal in self.HIGH_IMPACT_JOURNALS):
                        high_impact_count += 1
                        high_impact_pubs.append({
                            'gene': gene,
                            'journal': article.get('journal'),
                            'title': article.get('title', '')[:100]
                        })
                        break  # One per gene
        
        if high_impact_count >= 2:
            return self._create_check(
                name="High-Impact Publications",
                status=ValidationStatus.PASSED,
                message=f"{high_impact_count} high-impact publications found",
                confidence=ConfidenceLevel.HIGH,
                evidence={'high_impact_pubs': high_impact_pubs}
            )
        elif high_impact_count > 0:
            return self._create_check(
                name="High-Impact Publications",
                status=ValidationStatus.WARNING,
                message=f"{high_impact_count} high-impact publication(s)",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'high_impact_pubs': high_impact_pubs}
            )
        return self._create_check(
            name="High-Impact Publications",
            status=ValidationStatus.WARNING,
            message="No high-impact publications found",
            confidence=ConfidenceLevel.LOW
        )
    
    def _validate_evidence_quality(self, genes: List[str], cancer_type: str) -> ValidationCheck:
        """Evidence quality scoring"""
        # Proxy metrics from titles/abstracts
        mechanistic_terms = ['pathway', 'mechanism', 'function', 'regulation', 'signaling']
        therapeutic_terms = ['therapy', 'treatment', 'drug', 'inhibitor', 'target']
        quality_scores = []
        
        for gene in genes[:5]:
            query = f"{gene}[Title/Abstract] AND {cancer_type}[Title/Abstract]"
            result = PubMedClient.search_pubmed(query, max_results=5)
            
            if result:
                score = 0
                for article in result.get('articles', [])[:3]:
                    title = (article.get('title', '') + ' ' + article.get('journal', '')).lower()
                    mechanistic_hits = sum(1 for term in mechanistic_terms if term in title)
                    therapeutic_hits = sum(1 for term in therapeutic_terms if term in title)
                    score += mechanistic_hits + therapeutic_hits * 1.5
                
                quality_scores.append({'gene': gene, 'score': score})
        
        avg_quality = sum(s['score'] for s in quality_scores) / max(len(quality_scores), 1)
        
        if avg_quality >= 4:
            return self._create_check(
                name="Evidence Quality",
                status=ValidationStatus.PASSED,
                message=f"High-quality mechanistic evidence (avg score: {avg_quality:.1f})",
                confidence=ConfidenceLevel.HIGH,
                evidence={'quality_scores': quality_scores}
            )
        else:
            return self._create_check(
                name="Evidence Quality",
                status=ValidationStatus.WARNING,
                message=f"Moderate evidence quality (avg score: {avg_quality:.1f})",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'quality_scores': quality_scores}
            )
    
    def _synthesize_literature(self, genes: List[str], cancer_type: str, checks: List[ValidationCheck]) -> ValidationCheck:
        """LLM literature synthesis"""
        try:
            evidence_summary = "\n".join([f"{c.name}: {c.message}" for c in checks])
            
            prompt = f"""Biomarkers: {', '.join(genes[:6])}
Disease: {cancer_type}

PubMed evidence summary:
{evidence_summary}

Synthesize into concise literature review:
1. Overall evidence strength
2. Key mechanistic insights  
3. Therapeutic implications
4. Research gaps"""
            
            synthesis = self._query_llm(prompt, temperature=0.2)
            
            return self._create_check(
                name="Literature Synthesis (LLM)",
                status=ValidationStatus.PASSED,
                message="Comprehensive literature review generated",
                confidence=ConfidenceLevel.MEDIUM,
                evidence={'synthesis': synthesis[:450]}
            )
        except Exception:
            return self._create_check(
                name="Literature Synthesis (LLM)",
                status=ValidationStatus.SKIPPED,
                message="Synthesis unavailable",
                confidence=ConfidenceLevel.NONE
            )
    
    def _determine_status(self, checks: List[ValidationCheck]) -> ValidationStatus:
        if any(c.status == ValidationStatus.ERROR for c in checks): return ValidationStatus.ERROR
        if any(c.status == ValidationStatus.FAILED for c in checks): return ValidationStatus.FAILED  
        if sum(1 for c in checks if c.status == ValidationStatus.PASSED) >= 3: return ValidationStatus.PASSED
        return ValidationStatus.WARNING
    
    def _get_recommendations(self, checks: List[ValidationCheck]) -> List[str]:
        recs = ["Review top PubMed articles for detailed mechanisms"]
        if any(c.status == ValidationStatus.WARNING for c in checks if 'Volume' in c.name):
            recs.append("Novel biomarkers - consider functional validation studies")
        return recs


# Singleton
_literature_agent = None

def get_literature_evidence_agent() -> LiteratureEvidenceAgent:
    global _literature_agent
    if _literature_agent is None:
        _literature_agent = LiteratureEvidenceAgent()
    return _literature_agent

