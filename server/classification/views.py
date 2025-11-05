import os
import uuid
import logging
import base64
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.utils import timezone
from django.db import transaction
from .models import Patient, AnalysisSession, ClassificationResult, ModelPerformance
from .ml_service import ml_service
from .colorectal_cancer import colorectal_cancer_service
from .liver_cancer import liver_cancer_service
import shap

logger = logging.getLogger(__name__)

def align_patient_vector(df, feature_names):
    """
    Align single patient input vector from CSV df (gene rows) to match model's expected features.
    Missing genes have zero values.
    Returns a numpy array reshaped to (1, n_features).
    """
    series = df.iloc[:, 0]
    values = [series.get(g, 0.0) for g in feature_names]
    values_array = np.array(values).reshape(1, -1)  # Reshape to (1, n)
    return values_array

def generate_shap_bar_plot(model, df, feature_names, top_n=20):
    try:
        X = align_patient_vector(df, feature_names)
        explainer = shap.KernelExplainer(model.predict_proba, X)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_vals = shap_values[0]

        # Sort features by absolute SHAP value
        abs_shap = pd.Series(abs(shap_vals), index=feature_names)
        top_features = abs_shap.nlargest(top_n).index
        shap_vals_top = pd.Series(shap_vals, index=feature_names).loc[top_features]

        colors = ['#e74c3c' if v > 0 else '#3498db' for v in shap_vals_top]

        plt.switch_backend('Agg')
        plt.figure(figsize=(8, max(5, 0.4 * len(top_features))))
        shap_vals_top.sort_values().plot.barh(color=colors)
        plt.xlabel('SHAP value (feature impact)')
        plt.title(f'Top {top_n} Gene Contributions to Prediction')
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()
    except Exception as e:
        logger.error(f'SHAP bar plot generation failed: {e}')
        raise

@csrf_exempt
@require_http_methods(["POST"])
def analyze_classification(request):
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        uploaded_file = request.FILES['file']
        if not uploaded_file.name.endswith('.csv'):
            return JsonResponse({'error': 'Only CSV files are allowed'}, status=400)

        model_type = request.POST.get('model_type', 'lung_cancer')
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        patient_id = request.POST.get('patient_id', f'PAT-{uuid.uuid4().hex[:8].upper()}')

        if not age or not gender:
            return JsonResponse({'error': 'Age and gender are required'}, status=400)

        try:
            age = int(age)
            if age < 0 or age > 150:
                return JsonResponse({'error': 'Invalid age'}, status=400)
        except ValueError:
            return JsonResponse({'error': 'Age must be a number'}, status=400)

        if model_type == 'lung_cancer':
            service = ml_service
        elif model_type == 'colorectal_cancer':
            service = colorectal_cancer_service
        elif model_type == 'liver_cancer':
            service = liver_cancer_service
        else:
            return JsonResponse({'error': f'Unknown model type {model_type}'}, status=400)

        if not service.model_loaded:
            return JsonResponse({'error': 'ML model not loaded.'}, status=500)

        file_path = f'uploads/{patient_id}_{uploaded_file.name}'
        saved_path = default_storage.save(file_path, uploaded_file)

        with transaction.atomic():
            patient, created = Patient.objects.get_or_create(
                patient_id=patient_id,
                defaults={'age': age, 'gender': gender}
            )
            session = AnalysisSession.objects.create(
                patient=patient,
                model_type=model_type,
                file_name=uploaded_file.name,
                file_path=saved_path,
                file_size=uploaded_file.size,
                status='processing'
            )
            try:
                full_path = default_storage.path(saved_path)
                df = pd.read_csv(full_path, index_col=0)

                if df.empty:
                    raise ValueError("CSV file is empty")

                if model_type in ('colorectal_cancer', 'liver_cancer'):
                    processed_input = service.preprocess_patient_data(df)
                    predicted_class, predicted_prob = service.predict(processed_input)
                    individual_results = service.format_results(predicted_class, predicted_prob)
                    model_performance = []
                else:
                    data_array, gene_names = service.preprocess_rna_seq_data(df)
                    predicted_classes, probabilities = service.predict(data_array)
                    individual_results = service.format_classification_results(
                        predicted_classes, probabilities, gene_names, data_array.flatten()
                    )
                    model_performance = service.calculate_model_performance()

                ClassificationResult.objects.create(
                    analysis_session=session,
                    result_type=f'{model_type}_classification',
                    class_label=individual_results['patient_prediction']['label'],
                    probability=individual_results['patient_prediction']['confidence'],
                    confidence_score=1
                )

                for metric in model_performance:
                    ModelPerformance.objects.create(
                        analysis_session=session,
                        metric_name=metric['metric'],
                        metric_value=metric['value'],
                        metric_description=metric['description']
                    )

                session.status = 'completed'
                session.completed_at = timezone.now()
                session.save()

                if os.path.exists(full_path):
                    os.remove(full_path)

                return JsonResponse({
                    'session_id': str(session.session_id),
                    'patient_id': patient_id,
                    'classification_results': individual_results['patient_prediction'],
                    'model_performance': model_performance,
                    'gene_heatmap_data': individual_results.get('gene_heatmap_data'),
                    'status': 'success',
                })
            except Exception as e:
                session.status = 'failed'
                session.error_message = str(e)
                session.save()
                logger.error(f'Analysis failed: {e}')
                raise e

    except Exception as e:
        logger.error(f'Request processing failed: {e}')
        return JsonResponse({'error': f'Analysis failed: {e}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def generate_xai_graph(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    try:
        uploaded_file = request.FILES['file']
        if not uploaded_file.name.lower().endswith('.csv'):
            return JsonResponse({'error': 'Only CSV files are allowed'}, status=400)

        model_type = request.POST.get('model_type')
        if not model_type:
            return JsonResponse({'error': 'Model type is required'}, status=400)

        if model_type == 'lung_cancer':
            service = ml_service
        elif model_type == 'colorectal_cancer':
            service = colorectal_cancer_service
        elif model_type == 'liver_cancer':
            service = liver_cancer_service
        else:
            return JsonResponse({'error': 'Unknown model type'}, status=400)

        if not service.model_loaded:
            return JsonResponse({'error': 'Model not loaded'}, status=500)

        tmp_file_path = f'tmp/{uuid.uuid4()}.csv'
        saved_path = default_storage.save(tmp_file_path, uploaded_file)
        full_path = default_storage.path(saved_path)

        df = pd.read_csv(full_path, index_col=0)
        if df.empty:
            raise ValueError("CSV file is empty")

        feature_names = getattr(service, "feature_names", df.index.tolist())
        image_base64 = generate_shap_bar_plot(service.model, df, feature_names, top_n=20)

        if os.path.exists(full_path):
            os.remove(full_path)

        return JsonResponse({
            'image_base64': image_base64,
            'model_type': model_type,
            'status': 'success',
        })

    except Exception as e:
        logger.error(f'SHAP generation error: {e}')
        return JsonResponse({'error': f'Failed to generate SHAP graph: {str(e)}'}, status=500)

# Other existing views (get_results, get_analysis_history, export_report) remain unchanged and should be included as per previous versions.


@csrf_exempt
@require_http_methods(["POST"])
def analyze_classification(request):
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        uploaded_file = request.FILES['file']

        if not uploaded_file.name.endswith('.csv'):
            return JsonResponse({'error': 'Only CSV files are allowed'}, status=400)
        
        model_type = request.POST.get('model_type', 'lung_cancer')
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        patient_id = request.POST.get('patient_id', f'PAT-{uuid.uuid4().hex[:8].upper()}')

        if not age or not gender:
            return JsonResponse({'error': 'Age and gender are required'}, status=400)

        try:
            age = int(age)
            if age < 0 or age > 150:
                return JsonResponse({'error': 'Invalid age'}, status=400)
        except ValueError:
            return JsonResponse({'error': 'Age must be a number'}, status=400)

        # Select service based on model_type
        if model_type == 'lung_cancer':
            service = ml_service
        elif model_type == 'colorectal_cancer':
            service = colorectal_cancer_service
        elif model_type == 'liver_cancer':
            service = liver_cancer_service
        else:
            return JsonResponse({'error': f'Unknown model type: {model_type}'}, status=400)

        if not service.model_loaded:
            return JsonResponse({'error': 'ML model not loaded.'}, status=500)

        file_path = f'uploads/{patient_id}_{uploaded_file.name}'
        saved_file_path = default_storage.save(file_path, uploaded_file)

        with transaction.atomic():
            patient, created = Patient.objects.get_or_create(
                patient_id=patient_id,
                defaults={'age': age, 'gender': gender}
            )

            session = AnalysisSession.objects.create(
                patient=patient,
                model_type=model_type,
                file_name=uploaded_file.name,
                file_path=saved_file_path,
                file_size=uploaded_file.size,
                status='processing'
            )

            try:
                full_path = default_storage.path(saved_file_path)
                df = pd.read_csv(full_path, index_col=0)

                if df.empty:
                    raise ValueError("CSV file is empty")

                if model_type == 'colorectal_cancer' or model_type == 'liver_cancer':
                    preprocessed_data = service.preprocess_patient_data(df)
                    predicted_class, predicted_prob = service.predict(preprocessed_data)
                    individual_results = service.format_results(predicted_class, predicted_prob)
                    model_performance = []
                else:
                    data_array, gene_names = service.preprocess_rna_seq_data(df)
                    predicted_classes, probabilities = service.predict(data_array)
                    individual_results = service.format_classification_results(
                        predicted_classes, probabilities, gene_names, data_array.flatten()
                    )
                    model_performance = service.calculate_model_performance()

                ClassificationResult.objects.create(
                    analysis_session=session,
                    result_type=f'{model_type}_classification',
                    class_label=individual_results['patient_prediction']['label'],
                    probability=individual_results['patient_prediction']['confidence'],
                    confidence_score=1
                )

                for metric in model_performance:
                    ModelPerformance.objects.create(
                        analysis_session=session,
                        metric_name=metric['metric'],
                        metric_value=metric['value'],
                        metric_description=metric['description']
                    )

                session.status = 'completed'
                session.completed_at = timezone.now()
                session.save()

                if os.path.exists(full_path):
                    os.remove(full_path)

                response_data = {
                    'session_id': str(session.session_id),
                    'patient_id': patient_id,
                    'classification_results': individual_results['patient_prediction'],
                    'model_performance': model_performance,
                    'gene_heatmap_data': individual_results.get('gene_heatmap_data'),
                    'status': 'success',
                }
                return JsonResponse(response_data)

            except Exception as e:
                session.status = 'failed'
                session.error_message = str(e)
                session.save()
                logger.error(f"Analysis failed: {str(e)}")
                raise e

    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        return JsonResponse({'error': f'Analysis failed: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def get_results(request, session_id):
    try:
        session = AnalysisSession.objects.get(session_id=session_id)

        classification_results = []
        for result in session.results.all():
            classification_results.append({
                'subtype': result.class_label,
                'probability': result.probability,
                'count': int(result.confidence_score),
                'color': 'bg-red-500' if result.class_label == 'Cancer' else 'bg-green-500'
            })

        model_performance = []
        for metric in session.performance_metrics.all():
            model_performance.append({
                'metric': metric.metric_name,
                'value': metric.metric_value,
                'description': metric.metric_description
            })

        return JsonResponse({
            'session_id': str(session.session_id),
            'patient_id': session.patient.patient_id,
            'classification_results': classification_results,
            'model_performance': model_performance,
            'status': session.status,
            'created_at': session.created_at.isoformat(),
            'model_type': session.model_type
        })

    except AnalysisSession.DoesNotExist:
        return JsonResponse({'error': 'Analysis session not found'}, status=404)
    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}")
        return JsonResponse({'error': f'Failed to retrieve results: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def get_analysis_history(request):
    try:
        sessions = AnalysisSession.objects.select_related('patient').all().order_by('-created_at')[:20]

        history = []
        for session in sessions:
            cancer_count = session.results.filter(class_label='Cancer').first()
            normal_count = session.results.filter(class_label='Normal').first()

            history.append({
                'session_id': str(session.session_id),
                'patient_id': session.patient.patient_id,
                'model_type': session.model_type,
                'status': session.status,
                'created_at': session.created_at.isoformat(),
                'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                'cancer_count': int(cancer_count.confidence_score) if cancer_count else 0,
                'normal_count': int(normal_count.confidence_score) if normal_count else 0
            })

        return JsonResponse({'history': history})

    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return JsonResponse({'error': f'Failed to retrieve history: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def export_report(request, session_id):
    try:
        session = AnalysisSession.objects.get(session_id=session_id)
        return JsonResponse({
            'message': 'PDF export functionality will be implemented',
            'session_id': str(session.session_id),
            'patient_id': session.patient.patient_id,
            'status': session.status
        })

    except AnalysisSession.DoesNotExist:
        return JsonResponse({'error': 'Analysis session not found'}, status=404)
    except Exception as e:
        logger.error(f"Error in export: {str(e)}")
        return JsonResponse({'error': f'Export failed: {str(e)}'}, status=500)
