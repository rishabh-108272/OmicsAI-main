"""
Test script for Multi-Agentic AI Validation System
Demonstrates how to use each validation agent
"""

import os
import sys
import json
import django

# Add the project path (server directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

# Now import the agents
from classification.agents import (
    get_classification_validator,
    get_biomarker_validator,
    get_drug_validator,
    get_protein_validator,
    get_orchestrator
)


def test_classification_validation():
    """Test classification validation agent"""
    print("\n" + "="*60)
    print("Testing Classification Validation Agent")
    print("="*60)
    
    validator = get_classification_validator()
    
    # Sample data
    data = {
        'model_type': 'lung_cancer',
        'predicted_class': 'LUAD',
        'confidence': 0.87,
        'gene_expression': {
            'EGFR': 2.5,
            'KRAS': 1.8,
            'TP53': -1.2,
            'ALK': 0.5,
            'ROS1': 0.3,
            'BRAF': 1.1,
            'MET': 1.5,
            'ERBB2': 2.0,
            'RB1': -0.8,
            'PTEN': -1.0
        },
        'top_genes': ['EGFR', 'KRAS', 'TP53', 'ALK', 'BRAF', 'MET', 'ERBB2'],
        'patient_id': 'TEST-001',
        'model_performance': {
            'accuracy': 0.92,
            'f1_score': 0.89,
            'auc': 0.95
        }
    }
    
    result = validator.validate(data)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    
    return result


def test_biomarker_validation():
    """Test biomarker validation agent"""
    print("\n" + "="*60)
    print("Testing Biomarker Validation Agent")
    print("="*60)
    
    validator = get_biomarker_validator()
    
    # Sample data
    data = {
        'cancer_type': 'lung_cancer',
        'biomarkers': [
            {'gene': 'EGFR', 'importance': 0.95, 'p_value': 0.001},
            {'gene': 'KRAS', 'importance': 0.89, 'p_value': 0.003},
            {'gene': 'TP53', 'importance': 0.82, 'p_value': 0.01},
            {'gene': 'ALK', 'importance': 0.75, 'p_value': 0.02},
            {'gene': 'BRAF', 'importance': 0.68, 'p_value': 0.05},
            {'gene': 'MET', 'importance': 0.55, 'p_value': 0.08},
            {'gene': 'ERBB2', 'importance': 0.48, 'p_value': 0.1},
            {'gene': 'ROS1', 'importance': 0.42, 'p_value': 0.15},
            {'gene': 'FGFR1', 'importance': 0.35, 'p_value': 0.2},
            {'gene': 'PDGFRB', 'importance': 0.28, 'p_value': 0.25}
        ],
        'pathway_data': {},
        'heatmap_data': {}
    }
    
    result = validator.validate(data)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    
    return result


def test_drug_repurposing_validation():
    """Test drug repurposing validation agent"""
    print("\n" + "="*60)
    print("Testing Drug Repurposing Validation Agent")
    print("="*60)
    
    validator = get_drug_validator()
    
    # Sample data
    data = {
        'cancer_type': 'lung_cancer',
        'biomarkers': ['EGFR', 'KRAS', 'TP53', 'ALK', 'BRAF'],
        'candidates': [
            {
                'drug_name': 'Osimertinib',
                'target': 'EGFR',
                'hops_from_biomarker': 0,
                'score': 0.95,
                'evidence': 'Direct target'
            },
            {
                'drug_name': 'Erlotinib',
                'target': 'EGFR',
                'hops_from_biomarker': 0,
                'score': 0.88,
                'evidence': 'Direct target'
            },
            {
                'drug_name': 'Crizotinib',
                'target': 'ALK',
                'hops_from_biomarker': 0,
                'score': 0.85,
                'evidence': 'Direct target'
            },
            {
                'drug_name': 'Vemurafenib',
                'target': 'BRAF',
                'hops_from_biomarker': 0,
                'score': 0.82,
                'evidence': 'Direct target'
            },
            {
                'drug_name': 'Trametinib',
                'target': 'MAP2K1',
                'hops_from_biomarker': 1,
                'score': 0.65,
                'evidence': '1 hop from KRAS'
            },
            {
                'drug_name': 'Selumetinib',
                'target': 'MAP2K1',
                'hops_from_biomarker': 1,
                'score': 0.55,
                'evidence': '1 hop from KRAS'
            }
        ],
        'graph_data': {}
    }
    
    result = validator.validate(data)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    
    return result


def test_protein_validation():
    """Test protein structure validation agent"""
    print("\n" + "="*60)
    print("Testing Protein Structure Validation Agent")
    print("="*60)
    
    validator = get_protein_validator()
    
    # Sample data - EGFR-like protein
    import random
    random.seed(42)
    
    # Generate realistic pLDDT scores
    plddt_scores = []
    for i in range(650):
        if i < 50:  # N-terminus - low confidence (signal peptide)
            plddt_scores.append(random.uniform(30, 50))
        elif i < 200:  # Extracellular domain
            plddt_scores.append(random.uniform(70, 95))
        elif i < 230:  # Transmembrane
            plddt_scores.append(random.uniform(80, 95))
        else:  # Kinase domain - high confidence
            plddt_scores.append(random.uniform(85, 98))
    
    data = {
        'protein_id': 'P00533',
        'protein_name': 'EGFR',
        'sequence': 'MKSGLPSS星' * 65,  # Placeholder
        'plddt_scores': plddt_scores,
        'structure_data': {}
    }
    
    result = validator.validate(data)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    
    return result


def test_orchestrator():
    """Test the orchestrator with all agents"""
    print("\n" + "="*60)
    print("Testing Agent Orchestrator")
    print("="*60)
    
    orchestrator = get_orchestrator()
    
    # Test data that works with multiple agents
    data = {
        'model_type': 'lung_cancer',
        'predicted_class': 'LUAD',
        'confidence': 0.87,
        'gene_expression': {
            'EGFR': 2.5,
            'KRAS': 1.8,
            'TP53': -1.2,
            'ALK': 0.5
        },
        'top_genes': ['EGFR', 'KRAS', 'TP53', 'ALK', 'BRAF', 'MET'],
        'patient_id': 'TEST-001'
    }
    
    result = orchestrator.validate_all(data, agent_types=['classification'])
    print(json.dumps(result, indent=2, default=str))
    
    return result


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MULTI-AGENTIC AI VALIDATION SYSTEM TEST")
    print("="*60)
    
    # Check for Groq API key
    groq_key = os.environ.get('GROQ_API_KEY')
    if not groq_key:
        print("\n⚠️  WARNING: GROQ_API_KEY not set in environment!")
        print("   Set it with: export GROQ_API_KEY=your_key")
        print("   Or create a .env file in classification/agents/")
        print("\n   LLM-based validation will be skipped without an API key.\n")
    
    # Run tests
    try:
        # Test individual agents
        test_classification_validation()
        test_biomarker_validation()
        test_drug_repurposing_validation()
        test_protein_validation()
        
        # Test orchestrator
        test_orchestrator()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

