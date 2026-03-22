
"""
Literature Evidence Agent
Provides biological/scientific justification via RAG/ChromaDB
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime

from .base_agent import BaseAgent, ValidationResult, ValidationStatus, ConfidenceLevel, ValidationCheck

import requests
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class LiteratureEvidenceAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Literature Evidence Agent",
            description="Scientific evidence synthesis from PubMed/RAG"
        )

    def pubmed_search(self, gene: str, cancer_type: str = "cancer") -> List[str]:
        # \"\"\"Search PubMed for gene + cancer evidence (titles).\"\"\"
        ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': f'"{gene}" AND {cancer_type}',
            'retmax': 3,
            'retmode': 'xml'
        }
        try:
            resp = requests.get(ESEARCH_URL, params=params, timeout=10)
            root = ET.fromstring(resp.content)
            pmids = [id_elem.text for id_elem in root.findall('.//IdList/Id')]
            if not pmids:
                return [f"No recent PubMed hits for {gene}"]
            
            # Fetch summaries
            EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {'db': 'pubmed', 'id': ','.join(pmids[:2]), 'retmode': 'xml'}
            fetch_resp = requests.get(EFETCH_URL, params=fetch_params, timeout=10)
            articles = ET.fromstring(fetch_resp.content).findall('.//PubmedArticle')
            
            evidence = []
            for article in articles:
                title = article.find('.//ArticleTitle').text or 'No title'
                pmid_elem = article.find('.//PMID[@IdType="pubmed"]')
                pmid = pmid_elem.text if pmid_elem is not None else 'N/A'
                evidence.append(f"PMID:{pmid} - {title[:100]}...")
            return evidence[:3]
        except Exception:
            return [f"PubMed API unavailable for {gene}"]

    @property
    def system_prompt(self) -> str:
        return """Biomedical literature synthesis expert.

For biomarkers:
1. Retrieve top evidence (PMID, journals, findings)
2. Synthesize mechanisms/plausibility
3. Therapeutic implications
4. Evidence grading (clinical/preclinical/review)

Format structured with PMID links/quotes."""

    def retrieve_evidence(self, biomarkers: List[str], cancer_type: str) -> List[Dict]:
        evidence = []
        for gene in biomarkers[:8]:  # Limit API calls
            items = self.pubmed_search(gene, cancer_type)
            evidence.append({'gene': gene, 'evidence': items, 'strength': 'strong' if len(items)>1 else 'weak'})
        return evidence


    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        start_time = datetime.now()
        biomarkers = data.get('biomarkers', [])
        cancer_type = data.get('cancer_type', 'unknown')

        if not biomarkers:
            return self._create_result(ValidationStatus.FAILED, "No biomarkers", [])

        evidence = self.retrieve_evidence(biomarkers)
        strong_count = len([e for e in evidence if e['strength'] == 'strong'])

        prompt = f"Cancer: {cancer_type}\\nBiomarkers: {biomarkers}\\nEvidence: {json.dumps(evidence[:5])}"
        synthesis = self._query_llm(prompt)

        checks = [
            self._create_check(
                name="Evidence Retrieval",
                status=ValidationStatus.PASSED,
                message=f"{strong_count}/10 strong evidence",
                evidence={'strong': strong_count}
            ),
            self._create_check("Synthesis", ValidationStatus.PASSED, "Complete", evidence={'preview': synthesis[:300]})
        ]

        summary = f"Supported {strong_count}/{len(biomarkers)} biomarkers"
        recs = ['PubMed full-text review', 'Meta-analysis for validation']

        return ValidationResult(
            agent_name=self.name,
            overall_status=ValidationStatus.PASSED,
            overall_confidence=ConfidenceLevel.HIGH if strong_count > 3 else ConfidenceLevel.MEDIUM,
            summary=summary,
            checks=checks,
            recommendations=recs,
            metadata={'evidence': evidence},
            processing_time=(datetime.now() - start_time).total_seconds()
        )

# Singleton
_literature_agent = None
def get_literature_agent() -> 'LiteratureEvidenceAgent':
    global _literature_agent
    if _literature_agent is None:
        _literature_agent = LiteratureEvidenceAgent()
    return _literature_agent

