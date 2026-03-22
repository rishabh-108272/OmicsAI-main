from django.urls import path
from . import views
from . import validation_views

app_name = 'classification'

urlpatterns = [
    path('classification/analyze/', views.analyze_classification, name='analyze'),
    path('classification/results/<uuid:session_id>/', views.get_results, name='get_results'),
    path('classification/export/<uuid:session_id>/', views.export_report, name='export_report'),
    path('classification/history/', views.get_analysis_history, name='history'),

    path('classification/xai/graph/', views.generate_xai_graph, name='generate_xai_graph'),
    path('classification/drug-repurposing/', views.drug_repurposing_engine, name='drug_repurposing'),
    path('classification/ai-agent/', views.multi_agent_rag_view, name='ai_agent'),
    
    # Multi-Agentic AI Validation Endpoints
    path('validation/classification/', validation_views.validate_classification, name='validate_classification'),
    path('validation/biomarkers/', validation_views.validate_biomarkers, name='validate_biomarkers'),
    path('validation/drug-repurposing/', validation_views.validate_drug_repurposing, name='validate_drug_repurposing'),
    path('validation/protein/', validation_views.validate_protein_structure, name='validate_protein'),
    path('validation/all/', validation_views.validate_all, name='validate_all'),
    path('validation/agents/', validation_views.get_validation_agents, name='get_validation_agents'),
]
