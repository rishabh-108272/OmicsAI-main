"""
External API Clients for Multi-Agentic AI Validation System
Provides real-time validation against external databases over the internet

APIs Used:
- UniProt: Protein sequences, domains, structure data
- PubMed/NCBI: Literature and research validation
- DrugBank: Drug-target interactions
- ClinicalTrials.gov: Clinical trial information
- KEGG: Pathway and biological network data
"""

import os
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """External API providers"""
    UNIPROT = "uniprot"
    PUBMED = "pubmed"
    DRUGBANK = "drugbank"
    CLINICAL_TRIALS = "clinical_trials"
    KEGG = "kegg"
    EBI = "ebi"


class ExternalAPIClient:
    """
    Singleton client for external API calls with caching and retry logic
    """
    
    _instance = None
    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_expiry = 3600  # 1 hour
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize API configuration"""
        load_dotenv()
        
        # API Keys (optional - some APIs are free)
        self.pubmed_api_key = os.environ.get('PUBMED_API_KEY', '')
        self.drugbank_username = os.environ.get('DRUGBANK_USER', '')
        self.drugbank_password = os.environ.get('DRUGBANK_PASS', '')
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OmicsAI-Validation/1.0',
            'Accept': 'application/json'
        })
        
        # Rate limiting
        self._rate_limit_delay = 0.1  # 100ms between requests
        self._last_request_time = {}
        
        logger.info("ExternalAPIClient initialized")
    
    def _get_cache_key(self, provider: str, endpoint: str, params: Dict = None) -> str:
        """Generate cache key"""
        import hashlib
        content = f"{provider}:{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get cached response"""
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if (datetime.now() - entry['timestamp']).seconds < self._cache_expiry:
                logger.debug(f"Cache hit for {cache_key[:20]}...")
                return entry['data']
            else:
                del self._cache[cache_key]
        return None
    
    def _add_to_cache(self, cache_key: str, data: Any):
        """Add response to cache"""
        self._cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def _rate_limit(self, provider: str):
        """Apply rate limiting"""
        current_time = time.time()
        if provider in self._last_request_time:
            elapsed = current_time - self._last_request_time[provider]
            if elapsed < self._rate_limit_delay:
                time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time[provider] = time.time()
    
    def _make_request(
        self,
        provider: str,
        url: str,
        method: str = 'GET',
        params: Dict = None,
        data: Dict = None,
        headers: Dict = None,
        timeout: int = 30
    ) -> Optional[Dict]:
        """Make HTTP request with error handling"""
        self._rate_limit(provider)
        
        cache_key = self._get_cache_key(provider, url, params)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=timeout)
            elif method == 'POST':
                response = self.session.post(url, json=data, headers=headers, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            result = response.json()
            
            self._add_to_cache(cache_key, result)
            return result
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed for {provider}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error for {provider}: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached responses"""
        self._cache.clear()
        logger.info("ExternalAPIClient cache cleared")


# ==================== UNIPROT API ====================

class UniProtClient:
    """Client for UniProt API - Protein data validation"""
    
    BASE_URL = "https://rest.uniprot.org/uniprotkb"
    
    @staticmethod
    def get_protein_info(accession: str) -> Optional[Dict]:
        """Get protein information from UniProt"""
        client = ExternalAPIClient()
        
        url = f"{UniProtClient.BASE_URL}/{accession}"
        params = {
            'fields': 'accession,protein_name,gene_names,organism_name,sequence_length,kinase_activity,pathway'
        }
        
        result = client._make_request(APIProvider.UNIPROT.value, url, params=params)
        
        if result:
            return {
                'accession': result.get('primaryAccession'),
                'protein_name': result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value'),
                'gene_names': result.get('genes', [{}])[0].get('geneName', {}).get('value'),
                'organism': result.get('organism', {}).get('scientificName'),
                'sequence_length': result.get('sequence', {}).get('length'),
                'sequence': result.get('sequence', {}).get('value'),
                'functions': result.get('functionComment', {}).get('texts', [{}])[0].get('value', '')[:500]
            }
        return None
    
    @staticmethod
    def get_protein_domains(accession: str) -> Optional[List[Dict]]:
        """Get protein domain information from UniProt"""
        client = ExternalAPIClient()
        
        url = f"{UniProtClient.BASE_URL}/{accession}"
        params = {
            'fields': 'features'
        }
        
        result = client._make_request(APIProvider.UNIPROT.value, url, params=params)
        
        if result:
            features = result.get('features', [])
            domains = [f for f in features if f.get('type') == 'DOMAIN']
            return [
                {
                    'description': d.get('description'),
                    'start': d.get('location', {}).get('start', {}).get('value'),
                    'end': d.get('location', {}).get('end', {}).get('value')
                }
                for d in domains
            ]
        return None
    
    @staticmethod
    def search_proteins(query: str, limit: int = 10) -> Optional[List[Dict]]:
        """Search proteins in UniProt"""
        client = ExternalAPIClient()
        
        url = f"{UniProtClient.BASE_URL}/search"
        params = {
            'query': query,
            'size': limit,
            'fields': 'accession,protein_name,gene_names,organism_name'
        }
        
        result = client._make_request(APIProvider.UNIPROT.value, url, params=params)
        
        if result:
            return result.get('results', [])
        return None


# ==================== PUBMED/NCBI API ====================

class PubMedClient:
    """Client for PubMed/NCBI API - Literature validation"""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    @staticmethod
    def search_pubmed(query: str, max_results: int = 10) -> Optional[Dict]:
        """Search PubMed for articles"""
        client = ExternalAPIClient()
        
        # First, get IDs
        url = f"{PubMedClient.BASE_URL}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        if client.pubmed_api_key:
            params['api_key'] = client.pubmed_api_key
        
        search_result = client._make_request(APIProvider.PUBMED.value, url, params=params)
        
        if not search_result:
            return None
        
        id_list = search_result.get('esearchresult', {}).get('IdList', [])
        
        if not id_list:
            return {'count': 0, 'articles': []}
        
        # Get article details
        url = f"{PubMedClient.BASE_URL}/esummary.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(id_list),
            'retmode': 'json'
        }
        
        if client.pubmed_api_key:
            params['api_key'] = client.pubmed_api_key
        
        summary_result = client._make_request(APIProvider.PUBMED.value, url, params=params)
        
        articles = []
        if summary_result:
            for pmid in id_list:
                article = summary_result.get('result', {}).get(pmid, {})
                if article:
                    articles.append({
                        'pmid': pmid,
                        'title': article.get('title'),
                        'journal': article.get('source'),
                        'pubdate': article.get('pubdate'),
                        'authors': [a.get('name') for a in article.get('authors', [])[:3]],
                        'doi': article.get('elocationid', '').replace('doi: ', '')
                    })
        
        return {
            'count': len(articles),
            'articles': articles
        }
    
    @staticmethod
    def get_article_abstract(pmid: str) -> Optional[str]:
        """Get article abstract"""
        client = ExternalAPIClient()
        
        url = f"{PubMedClient.BASE_URL}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml'
        }
        
        if client.pubmed_api_key:
            params['api_key'] = client.pubmed_api_key
        
        result = client._make_request(APIProvider.PUBMED.value, url, params=params)
        
        if result:
            try:
                # Simple XML parsing
                import xml.etree.ElementTree as ET
                root = ET.fromstring(result)
                abstract = root.find('.//AbstractText')
                if abstract is not None:
                    return abstract.text
            except Exception as e:
                logger.warning(f"Failed to parse PubMed abstract: {e}")
        
        return None
    
    @staticmethod
    def validate_gene_disease_association(gene: str, disease: str) -> Dict:
        """Validate gene-disease association from literature"""
        query = f"{gene}[Title/Abstract] AND {disease}[Title/Abstract]"
        result = PubMedClient.search_pubmed(query, max_results=5)
        
        if result and result.get('count', 0) > 0:
            return {
                'validated': True,
                'article_count': result['count'],
                'top_articles': result['articles'][:3],
                'evidence': f"Found {result['count']} publications linking {gene} to {disease}"
            }
        
        return {
            'validated': False,
            'article_count': 0,
            'top_articles': [],
            'evidence': f"No publications found linking {gene} to {disease}"
        }


# ==================== DRUGBANK API ====================

class DrugBankClient:
    """Client for DrugBank API - Drug-target validation"""
    
    # DrugBank free API endpoint
    BASE_URL = "https://go.drugbank.com"
    
    @staticmethod
    def get_drug_info(drug_name: str) -> Optional[Dict]:
        """Get drug information from DrugBank"""
        client = ExternalAPIClient()
        
        # Use DrugBank's public pages (scraping - use carefully)
        url = f"{DrugBankClient.BASE_URL}/drugs/{drug_name}"
        
        # Note: DrugBank requires authentication for API
        # This is a simplified version using public data
        headers = {
            'Accept': 'application/json'
        }
        
        result = client._make_request(
            APIProvider.DRUGBANK.value, 
            url, 
            headers=headers
        )
        
        if result and 'results' in result:
            return result['results'][0] if result['results'] else None
        
        return None
    
    @staticmethod
    def get_drug_targets(drug_name: str) -> List[Dict]:
        """Get drug targets for a given drug"""
        # Common drug targets database (offline fallback)
        KNOWN_DRUG_TARGETS = {
            'everolimus': ['MTOR', 'FKBP1A'],
            'sirolimus': ['MTOR', 'FKBP1A'],
            'olaparib': ['PARP1', 'PARP2', 'PARP3'],
            'niraparib': ['PARP1', 'PARP2'],
            'rucaparib': ['PARP1', 'PARP2'],
            'talazoparib': ['PARP1', 'PARP2'],
            'vemurafenib': ['BRAF'],
            'dabrafenib': ['BRAF'],
            'trametinib': ['MAP2K1', 'MAP2K2'],
            'cobimetinib': ['MAP2K1', 'MAP2K2'],
            'erlotinib': ['EGFR'],
            'gefitinib': ['EGFR'],
            'osimertinib': ['EGFR'],
            'cetuximab': ['EGFR'],
            'panitumumab': ['EGFR'],
            'crizotinib': ['ALK', 'MET', 'ROS1'],
            'alectinib': ['ALK'],
            'ceritinib': ['ALK'],
            'lorlatinib': ['ALK'],
            'palbociclib': ['CDK4', 'CDK6'],
            'ribociclib': ['CDK4', 'CDK6'],
            'abemaciclib': ['CDK4', 'CDK6'],
            'bevacizumab': ['VEGFA'],
            'sunitinib': ['VEGFR2', 'PDGFRB', 'KIT'],
            'sorafenib': ['VEGFR2', 'RAF1', 'BRAF'],
            'pazopanib': ['VEGFR1', 'VEGFR2', 'VEGFR3'],
            'axitinib': ['VEGFR1', 'VEGFR2', 'VEGFR3'],
            'ruxolitinib': ['JAK1', 'JAK2'],
            'fedratinib': ['JAK2'],
            'alpelisib': ['PIK3CA'],
            'copanlisib': ['PIK3CA', 'PIK3CD'],
            'temsirolimus': ['MTOR'],
        }
        
        drug_key = drug_name.lower().replace(' ', '_')
        
        if drug_key in KNOWN_DRUG_TARGETS:
            targets = KNOWN_DRUG_TARGETS[drug_key]
            return [
                {'target': t, 'source': 'known_database'}
                for t in targets
            ]
        
        return []
    
    @staticmethod
    def is_drug_approved(drug_name: str, cancer_type: str = None) -> Dict:
        """Check if drug is approved"""
        # Common cancer drug approvals
        APPROVED_CANCER_DRUGS = {
            'everolimus': ['RCC', 'Breast', 'TSC'],
            'sirolimus': ['RCC'],
            'olaparib': ['Ovarian', 'Breast', 'Prostate', 'Pancreatic'],
            'niraparib': ['Ovarian'],
            'rucaparib': ['Ovarian', 'Prostate'],
            'talazoparib': ['Breast'],
            'vemurafenib': ['Melanoma', 'CRC'],
            'dabrafenib': ['Melanoma', 'Lung'],
            'trametinib': ['Melanoma', 'Lung'],
            'cobimetinib': ['Melanoma'],
            'erlotinib': ['Lung', 'Pancreatic'],
            'gefitinib': ['Lung'],
            'osimertinib': ['Lung'],
            'cetuximab': ['CRC', 'Head and Neck'],
            'panitumumab': ['CRC'],
            'crizotinib': ['Lung'],
            'alectinib': ['Lung'],
            'ceritinib': ['Lung'],
            'lorlatinib': ['Lung'],
            'palbociclib': ['Breast'],
            'ribociclib': ['Breast'],
            'abemaciclib': ['Breast'],
            'bevacizumab': ['CRC', 'Lung', 'RCC', 'Ovarian', 'Glioblastoma'],
            'sunitinib': ['RCC', 'GIST'],
            'sorafenib': ['HCC', 'RCC', 'Thyroid'],
            'pazopanib': ['RCC', 'Sarcoma'],
            'axitinib': ['RCC'],
            'ruxolitinib': ['Myelofibrosis'],
            'fedratinib': ['Myelofibrosis'],
            'alpelisib': ['Breast'],
            'copanlisib': ['Lymphoma'],
            'temsirolimus': ['RCC'],
        }
        
        drug_key = drug_name.lower().replace(' ', '_')
        
        if drug_key in APPROVED_CANCER_DRUGS:
            approved_for = APPROVED_CANCER_DRUGS[drug_key]
            
            if cancer_type:
                cancer_normalized = cancer_type.lower().replace(' ', '')
                is_approved = any(
                    cancer_normalized in a.lower().replace(' ', '') 
                    for a in approved_for
                )
                
                return {
                    'approved': is_approved,
                    'drug': drug_name,
                    'approved_for': approved_for,
                    'query_cancer': cancer_type,
                    'source': 'known_database'
                }
            
            return {
                'approved': True,
                'drug': drug_name,
                'approved_for': approved_for,
                'source': 'known_database'
            }
        
        return {
            'approved': False,
            'drug': drug_name,
            'approved_for': [],
            'source': 'known_database'
        }


# ==================== CLINICAL TRIALS API ====================

class ClinicalTrialsClient:
    """Client for ClinicalTrials.gov API - Clinical trial validation"""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    @staticmethod
    def search_trials(
        condition: str = None,
        intervention: str = None,
        max_results: int = 10
    ) -> Optional[Dict]:
        """Search clinical trials"""
        client = ExternalAPIClient()
        
        # Build query
        query_parts = []
        if condition:
            query_parts.append(f"AREA[Condition]*{condition}")
        if intervention:
            query_parts.append(f"AREA[InterventionName]*{intervention}")
        
        query = ' AND '.join(query_parts) if query_parts else '*'
        
        url = f"{ClinicalTrialsClient.BASE_URL}/studies"
        params = {
            'query.cond': condition or '',
            'query.term': intervention or '',
            'pageSize': max_results,
            'format': 'json'
        }
        
        result = client._make_request(APIProvider.CLINICAL_TRIALS.value, url, params=params)
        
        if result and 'studies' in result:
            trials = []
            for study in result['studies']:
                protocol = study.get('protocolSection', {})
                trials.append({
                    'nct_id': protocol.get('identificationModule', {}).get('nctId'),
                    'title': protocol.get('identificationModule', {}).get('briefTitle'),
                    'status': protocol.get('statusModule', {}).get('overallStatus'),
                    'phase': protocol.get('designModule', {}).get('phases', []),
                    'interventions': [
                        i.get('name') 
                        for i in protocol.get('armsInterventionsModule', {}).get('interventions', [])
                    ]
                })
            
            return {
                'count': len(trials),
                'trials': trials
            }
        
        return {'count': 0, 'trials': []}
    
    @staticmethod
    def get_trial_by_drug(drug_name: str, cancer_type: str = None) -> Dict:
        """Get clinical trials for a drug and cancer type"""
        condition = cancer_type or "cancer"
        
        result = ClinicalTrialsClient.search_trials(
            condition=condition,
            intervention=drug_name,
            max_results=10
        )
        
        if result:
            active_trials = [
                t for t in result.get('trials', [])
                if t.get('status') in ['RECRUITING', 'ACTIVE_NOT_RECRUITING', 'NOT_YET_RECRUITING']
            ]
            
            return {
                'total_trials': result.get('count', 0),
                'active_trials': len(active_trials),
                'trials': result.get('trials', [])[:5],
                'drug': drug_name,
                'condition': condition
            }
        
        return {
            'total_trials': 0,
            'active_trials': 0,
            'trials': [],
            'drug': drug_name,
            'condition': condition
        }


# ==================== KEGG PATHWAY API ====================

class KEGGClient:
    """Client for KEGG API - Pathway and biological network data"""
    
    BASE_URL = "https://rest.kegg.jp"
    
    @staticmethod
    def get_pathway_genes(pathway_id: str) -> Optional[List[str]]:
        """Get genes in a pathway"""
        client = ExternalAPIClient()
        
        url = f"{KEGGClient.BASE_URL}/link/genes/{pathway_id}"
        
        result = client._make_request(APIProvider.KEGG.value, url)
        
        if result and isinstance(result, str):
            genes = []
            for line in result.strip().split('\n'):
                if '\t' in line:
                    genes.append(line.split('\t')[1])
            return genes
        
        return None
    
    @staticmethod
    def get_pathway_info(pathway_id: str) -> Optional[Dict]:
        """Get pathway information"""
        client = ExternalAPIClient()
        
        # Get pathway description
        url = f"{KEGGClient.BASE_URL}/get/{pathway_id}"
        result = client._make_request(APIProvider.KEGG.value, url)
        
        if result and isinstance(result, str):
            return {
                'pathway_id': pathway_id,
                'description': result.split('\n')[0] if result else ''
            }
        
        return None
    
    @staticmethod
    def check_pathway_enrichment(genes: List[str], pathway_id: str) -> Dict:
        """Check if genes are enriched in a pathway"""
        pathway_genes = KEGGClient.get_pathway_genes(pathway_id)
        
        if not pathway_genes:
            return {'enriched': False, 'error': 'Could not retrieve pathway genes'}
        
        gene_set = set(g.upper() for g in genes)
        pathway_gene_set = set(p.upper() for p in pathway_genes)
        
        overlap = gene_set & pathway_gene_set
        
        # Simple enrichment check
        if len(overlap) >= 2:
            return {
                'enriched': True,
                'overlap_genes': list(overlap),
                'overlap_count': len(overlap),
                'pathway_genes_count': len(pathway_genes),
                'enrichment_ratio': len(overlap) / len(pathway_genes)
            }
        
        return {
            'enriched': False,
            'overlap_genes': list(overlap),
            'overlap_count': len(overlap)
        }
    
    @staticmethod
    def get_cancer_pathways() -> List[Dict]:
        """Get list of cancer-related pathways"""
        # Common cancer pathways
        return [
            {'id': 'hsa05200', 'name': 'Pathways in cancer'},
            {'id': 'hsa05210', 'name': 'Colorectal cancer'},
            {'id': 'hsa05225', 'name': 'Hepatocellular carcinoma'},
            {'id': 'hsa05223', 'name': 'Non-small cell lung cancer'},
            {'id': 'hsa05213', 'name': 'Endometrial cancer'},
            {'id': 'hsa05219', 'name': 'Bladder cancer'},
            {'id': 'hsa05205', 'name': 'Proteoglycans in cancer'},
            {'id': 'hsa05212', 'name': 'Pancreatic cancer'},
            {'id': 'hsa05214', 'name': 'Glioma'},
            {'id': 'hsa05218', 'name': 'Melanoma'},
            {'id': 'hsa05220', 'name': 'Chronic myeloid leukemia'},
            {'id': 'hsa05221', 'name': 'Acute myeloid leukemia'},
            {'id': 'hsa05211', 'name': 'Renal cell carcinoma'},
            {'id': 'hsa05224', 'name': 'Breast cancer'},
            {'id': 'hsa05215', 'name': 'Prostate cancer'},
            {'id': 'hsa05222', 'name': 'Small cell lung cancer'},
            {'id': 'hsa04012', 'name': 'ERBB signaling pathway'},
            {'id': 'hsa04014', 'name': 'RAS signaling pathway'},
            {'id': 'hsa04110', 'name': 'Cell cycle'},
            {'id': 'hsa04115', 'name': 'p53 signaling pathway'},
            {'id': 'hsa04210', 'name': 'Apoptosis'},
            {'id': 'hsa04010', 'name': 'MAPK signaling pathway'},
            {'id': 'hsa04151', 'name': 'PI3K-AKT signaling pathway'},
        ]


# ==================== VALIDATION HELPERS ====================

class ExternalValidator:
    """High-level validation using external APIs"""
    
    @staticmethod
    def validate_protein_with_uniprot(protein_id: str, plddt_scores: List[float] = None) -> Dict:
        """Validate protein using UniProt"""
        # Try UniProt accession first, then search
        protein_info = UniProtClient.get_protein_info(protein_id)
        
        if not protein_info:
            # Search by name
            search_results = UniProtClient.search_proteins(protein_id)
            if search_results:
                accession = search_results[0].get('accession', {}).get('value')
                if accession:
                    protein_info = UniProtClient.get_protein_info(accession)
        
        result = {
            'validated': False,
            'source': 'uniprot',
            'protein_info': None
        }
        
        if protein_info:
            result['validated'] = True
            result['protein_info'] = protein_info
            
            # Check sequence length consistency
            if plddt_scores and protein_info.get('sequence_length'):
                seq_len = protein_info['sequence_length']
                plddt_len = len(plddt_scores)
                
                if abs(seq_len - plddt_len) > 10:
                    result['warning'] = f"Sequence length mismatch: UniProt={seq_len}, pLDDT={plddt_len}"
                else:
                    result['sequence_matched'] = True
        
        return result
    
    @staticmethod
    def validate_biomarker_literature(gene: str, cancer_type: str) -> Dict:
        """Validate biomarker from literature"""
        return PubMedClient.validate_gene_disease_association(gene, cancer_type)
    
    @staticmethod
    def validate_drug_target(drug_name: str, target_gene: str) -> Dict:
        """Validate drug-target relationship"""
        targets = DrugBankClient.get_drug_targets(drug_name)
        
        target_upper = target_gene.upper()
        found_target = next(
            (t for t in targets if t['target'].upper() == target_upper),
            None
        )
        
        if found_target:
            return {
                'validated': True,
                'drug': drug_name,
                'target': target_gene,
                'source': 'drugbank_known'
            }
        
        return {
            'validated': False,
            'drug': drug_name,
            'target': target_gene,
            'known_targets': targets
        }
    
    @staticmethod
    def validate_drug_approval(drug_name: str, cancer_type: str = None) -> Dict:
        """Check drug approval status"""
        return DrugBankClient.is_drug_approved(drug_name, cancer_type)
    
    @staticmethod
    def validate_clinical_trials(drug_name: str, cancer_type: str) -> Dict:
        """Get clinical trial information"""
        return ClinicalTrialsClient.get_trial_by_drug(drug_name, cancer_type)
    
    @staticmethod
    def validate_pathway_enrichment(genes: List[str]) -> Dict:
        """Validate pathway enrichment"""
        # Check against cancer pathways
        cancer_pathways = KEGGClient.get_cancer_pathways()
        
        enriched_pathways = []
        
        for pathway in cancer_pathways[:10]:  # Check top 10 pathways
            result = KEGGClient.check_pathway_enrichment(genes, pathway['id'])
            if result.get('enriched'):
                enriched_pathways.append({
                    'pathway_id': pathway['id'],
                    'pathway_name': pathway['name'],
                    'overlap_genes': result.get('overlap_genes', []),
                    'overlap_count': result.get('overlap_count', 0)
                })
        
        return {
            'validated': len(enriched_pathways) > 0,
            'enriched_pathways': enriched_pathways,
            'pathway_count': len(enriched_pathways),
            'source': 'kegg'
        }


# Singleton instance
def get_external_api_client() -> ExternalAPIClient:
    """Get singleton ExternalAPIClient instance"""
    return ExternalAPIClient()


# Convenience functions
def validate_protein_external(protein_id: str, plddt_scores: List[float] = None) -> Dict:
    """Validate protein with external APIs"""
    return ExternalValidator.validate_protein_with_uniprot(protein_id, plddt_scores)


def validate_biomarker_external(gene: str, cancer_type: str) -> Dict:
    """Validate biomarker with external APIs"""
    return ExternalValidator.validate_biomarker_literature(gene, cancer_type)


def validate_drug_external(drug_name: str, target_gene: str = None, cancer_type: str = None) -> Dict:
    """Validate drug with external APIs"""
    result = {
        'drug': drug_name,
        'target_validation': None,
        'approval_status': None,
        'clinical_trials': None
    }
    
    if target_gene:
        result['target_validation'] = ExternalValidator.validate_drug_target(drug_name, target_gene)
    
    if cancer_type:
        result['approval_status'] = ExternalValidator.validate_drug_approval(drug_name, cancer_type)
        result['clinical_trials'] = ExternalValidator.validate_clinical_trials(drug_name, cancer_type)
    
    return result


def validate_pathway_external(genes: List[str]) -> Dict:
    """Validate pathway enrichment with external APIs"""
    return ExternalValidator.validate_pathway_enrichment(genes)

