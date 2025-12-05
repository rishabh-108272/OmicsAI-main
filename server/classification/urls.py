from django.urls import path
from . import views

app_name = 'classification'

urlpatterns = [
    path('classification/analyze/', views.analyze_classification, name='analyze'),
    path('classification/results/<uuid:session_id>/', views.get_results, name='get_results'),
    path('classification/export/<uuid:session_id>/', views.export_report, name='export_report'),
    path('classification/history/', views.get_analysis_history, name='history'),

    path('classification/xai/graph/', views.generate_xai_graph, name='generate_xai_graph'),
    path('classification/drug-repurposing/', views.drug_repurposing_engine, name='drug_repurposing'),
    path('classification/ai-agent/', views.ai-agent, name='ai_agent'),
]
