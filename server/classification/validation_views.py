"""
Validation API Views for Multi-Agentic AI Validation System
Provides endpoints for validating classification, biomarker, drug repurposing, and protein results
"""

import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator

from .agents import (
    get_orchestrator,
    ValidationStatus
)

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def validate_classification(request):
    """
    Validate cancer classification results
    
    Expected POST data:
    {
        'model_type': 'colorectal_cancer' | 'liver_cancer' | 'lung_cancer',
        'predicted_class': str,
        'confidence': float,
        'gene_expression': dict,
        'top_genes': list,
        'patient_id': str,
        'model_performance': dict (optional)
    }
    """
    try:
        body = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    if not body:
        return JsonResponse({'error': 'No data provided'}, status=400)
    
    return JsonResponse({'deprecated': 'Use /validate-all/ with new agents (pathway_reasoning, drug_association, literature_evidence)'}, status=410)


@csrf_exempt
@require_http_methods(["POST"])
def validate_biomarkers(request):
    """
    Validate biomarker discovery results
    
    Expected POST data:
    {
        'cancer_type': str,
        'biomarkers': [
            {'gene': str, 'importance': float, 'p_value': float, ...}
        ],
        'pathway_data': dict (optional),
        'heatmap_data': dict (optional)
    }
    """
    try:
        body = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    if not body:
        return JsonResponse({'error': 'No data provided'}, status=400)
    
    return JsonResponse({'deprecated': 'Use /validate-all/ with new agents'}, status=410)


@csrf_exempt
@require_http_methods(["POST"])
def validate_drug_repurposing(request):
    """
    Validate drug repurposing candidates
    
    Expected POST data:
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
        'graph_data': dict (optional)
    }
    """
    try:
        body = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    if not body:
        return JsonResponse({'error': 'No data provided'}, status=400)
    
    return JsonResponse({'deprecated': 'Use /validate-all/ with new agents'}, status=410)


@csrf_exempt
@require_http_methods(["POST"])
def validate_protein_structure(request):
    """
    Validate AlphaFold protein structure predictions
    
    Expected POST data:
    {
        'protein_id': str,  # UniProt accession
        'protein_name': str,
        'sequence': str,
        'plddt_scores': list,  # Per-residue pLDDT
        'pae_scores': list (optional),  # Per-residue PAE
        'structure_data': dict (optional)
    }
    """
    try:
        body = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    if not body:
        return JsonResponse({'error': 'No data provided'}, status=400)
    
    return JsonResponse({'deprecated': 'Use /validate-all/ with new agents'}, status=410)


@csrf_exempt
@require_http_methods(["POST"])
def validate_all(request):
    """
    Run all validation agents for comprehensive analysis
    
    Expected POST data:
    {
        'validation_type': 'classification' | 'biomarker' | 'drug' | 'protein' | 'all',
        'data': {...}  # Data to validate
    }
    """
    try:
        body = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    validation_type = body.get('validation_type', 'all')
    data = body.get('data', {})
    
    if not data:
        return JsonResponse({'error': 'No data provided'}, status=400)
    
    orchestrator = get_orchestrator()
    
    # Determine which agents to run
    if validation_type == 'all':
        agent_types = None
    elif validation_type in orchestrator.get_available_agents():
        agent_types = [validation_type]
    else:
        return JsonResponse({
            'error': f'Invalid validation type: {validation_type}',
            'available_types': orchestrator.get_available_agents()
        }, status=400)
    
    try:
        result = orchestrator.validate_all(data, agent_types)
        return JsonResponse(result, status=200)
    except Exception as e:
        logger.error(f"Orchestrator validation error: {e}")
        return JsonResponse({
            'error': f'Validation orchestrator failed: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def get_validation_agents(request):
    """
    Get list of available validation agents
    
    Returns:
    {
        'agents': [
            {'name': str, 'description': str},
            ...
        ]
    }
    """
    orchestrator = get_orchestrator()
    
    agents_info = []
    for agent_type in orchestrator.get_available_agents():
        agent = orchestrator.agents.get(agent_type)
        if agent:
            agents_info.append({
                'type': agent_type,
                'name': agent.name,
                'description': agent.description
            })
    
    return JsonResponse({
        'agents': agents_info,
        'count': len(agents_info)
    }, status=200)

