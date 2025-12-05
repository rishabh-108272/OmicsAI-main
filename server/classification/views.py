import os
import uuid
import logging
import base64
import json
import networkx as nx
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
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

DGIDB_GRAPHQL_URL = "https://dgidb.org/api/graphql"

DGIDB_GRAPHQL_QUERY = """
query DrugInteractions($genes: [String!]) {
  genes(names: $genes) {
    nodes {
      interactions {
        drug {
          name
          conceptId
        }
        interactionScore
        interactionTypes {
          type
          directionality
        }
        interactionAttributes {
          name
          value
        }
        publications {
          pmid
        }
        sources {
          sourceDbName
        }
      }
    }
  }
}
"""



def fetch_dgidb_drugs_via_graphql(genes):
    """
    Query DGIdb GraphQL API for drug-gene interactions.

    Returns:
        dict: {
            "GENE": [
                {
                  "drug_name": "...",
                  "concept_id": "...",
                  "score": ...,
                  "types": [...],
                  "publications": [...],
                  "sources": [...],
                },
                ...
            ]
        }
    """
    if not genes:
        return {}

    # Normalized, unique gene names
    unique_genes = sorted({str(g).strip() for g in genes if str(g).strip()})
    if not unique_genes:
        return {}

    payload = {
        "query": DGIDB_GRAPHQL_QUERY,
        "variables": {"genes": unique_genes}
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # ---- API call ----
    try:
        resp = requests.post(
            DGIDB_GRAPHQL_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=30,
        )
    except Exception as e:
        logger.error(f"DGIdb GraphQL network error: {e}")
        return {}

    # ---- Response validation ----
    if resp.status_code != 200:
        logger.warning(f"DGIdb returned {resp.status_code}: {resp.text[:200]}")
        return {}

    try:
        data = resp.json()
    except ValueError:
        logger.warning(f"DGIdb returned non-JSON response: {resp.text[:200]}")
        return {}

    if "errors" in data:
        logger.warning(f"DGIdb GraphQL errors: {data['errors']}")
        return {}

    # ---- Parsing ----
    # Expected shape:
    # data -> genes -> nodes[] -> interactions[]
    root = data.get("data", {}).get("genes", {})
    nodes = root.get("nodes", []) or []

    gene_to_drugs = {}

    for gene_node in nodes:
        interactions = gene_node.get("interactions", []) or []

        for inter in interactions:
            drug = inter.get("drug") or {}
            drug_name = drug.get("name")
            concept_id = drug.get("conceptId")

            if not drug_name:
                continue

            # Determine gene by conceptId mapping (DGIdb nodes array is ordered same as input genes)
            # We need to match back to gene names:
            # So we iterate in same order `unique_genes`
            for gene in unique_genes:
                if gene not in gene_to_drugs:
                    gene_to_drugs[gene] = []

            # Find the gene by matching interaction list index
            # DGIdb preserves order of input genes, so nodes align
            gene_index = nodes.index(gene_node)
            if gene_index < len(unique_genes):
                gene_name = unique_genes[gene_index].upper()
            else:
                continue

            gene_to_drugs[gene_name].append({
                "drug_name": drug_name,
                "concept_id": concept_id,
                "score": inter.get("interactionScore"),
                "types": [t.get("type") for t in inter.get("interactionTypes", [])],
                "publications": [p.get("pmid") for p in inter.get("publications", [])],
                "sources": [s.get("sourceDbName") for s in inter.get("sources", [])],
            })

    return gene_to_drugs



@csrf_exempt
@require_http_methods(["POST"])
def drug_repurposing_engine(request):
    """
    Build a PPI graph from uploaded CSV using NetworkX and run BFS
    from seed biomarkers to identify nearby nodes (candidate targets),
    then enrich those targets with drug information from DGIdb.
    """
    try:
        if 'ppi_file' not in request.FILES:
            return JsonResponse({'error': 'No PPI file uploaded (ppi_file required)'}, status=400)

        uploaded_file = request.FILES['ppi_file']

        if not uploaded_file.name.lower().endswith('.csv'):
            return JsonResponse({'error': 'Only CSV files are allowed'}, status=400)

        cancer_type = request.POST.get('cancer_type', 'LUAD_LUSC')

        biomarkers_raw = request.POST.get('biomarkers', '[]')
        try:
            biomarkers = json.loads(biomarkers_raw)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid biomarkers JSON'}, status=400)

        # Normalise biomarker list
        if isinstance(biomarkers, str):
            biomarkers = [b.strip() for b in biomarkers.split(',') if b.strip()]
        biomarkers = [str(b).strip() for b in biomarkers if str(b).strip()]

        if not biomarkers:
            return JsonResponse({'error': 'At least one biomarker is required'}, status=400)

        # Save file temporarily (consistent with your existing pattern)
        file_path = f'tmp/ppi_{uuid.uuid4().hex}_{uploaded_file.name}'
        saved_path = default_storage.save(file_path, uploaded_file)
        full_path = default_storage.path(saved_path)

        try:
            df = pd.read_csv(full_path)

            if df.empty:
                raise ValueError("PPI CSV file is empty")

            # --- Build NetworkX graph from CSV ---
            if df.shape[1] < 2:
                raise ValueError("PPI CSV must have at least two columns (source, target)")

            src_col = df.columns[0]
            tgt_col = df.columns[1]
            weight_col = df.columns[2] if df.shape[1] > 2 else None

            G = nx.Graph()

            for _, row in df.iterrows():
                src = str(row[src_col]).strip()
                tgt = str(row[tgt_col]).strip()

                if not src or not tgt or src == 'nan' or tgt == 'nan':
                    continue

                if weight_col is not None:
                    try:
                        weight = float(row[weight_col])
                    except (ValueError, TypeError):
                        weight = 1.0
                    G.add_edge(src, tgt, weight=weight)
                else:
                    G.add_edge(src, tgt)

            # --- BFS from biomarkers ---
            max_hops = 2  # you can parameterise this later if needed
            candidates_map = {}  # node -> best candidate info
            subgraph_nodes = set()

            for biomarker in biomarkers:
                if biomarker not in G:
                    continue

                subgraph_nodes.add(biomarker)

                lengths = nx.single_source_shortest_path_length(G, biomarker, cutoff=max_hops)

                for node, dist in lengths.items():
                    subgraph_nodes.add(node)
                    if node == biomarker:
                        continue

                    # simple scoring: closer nodes get higher scores
                    score = 1.0 / (dist + 1e-6)

                    existing = candidates_map.get(node)
                    if existing is None or score > existing['score']:
                        candidates_map[node] = {
                            'drug_name': None,  # will be filled after DGIdb query
                            'target': node,
                            'nearest_biomarker': biomarker,
                            'hops_from_biomarker': dist,
                            'score': score,
                            'evidence': f'Nearest biomarker: {biomarker}, shortest path length: {dist}'
                        }

            # --- Enrich candidate targets with DGIdb drug info ---
            MAX_DRUGS_PER_TARGET = 3
            target_genes = list(candidates_map.keys())
            dgidb_mapping = fetch_dgidb_drugs_via_graphql(target_genes)

            for gene, info in candidates_map.items():
                drugs_info = (
                    dgidb_mapping.get(gene)
                    or dgidb_mapping.get(gene.upper())
                    or []
                )
                
                drugs_info_sorted = sorted(
                    drugs_info,
                    key=lambda d: (d.get("score") or 0),
                    reverse=True,
                )
                
                top_drugs_info=drugs_info_sorted[:MAX_DRUGS_PER_TARGET]
                simple_names = [d["drug_name"] for d in top_drugs_info]

                info['drug_name'] = ", ".join(simple_names) if simple_names else None
                info['drug_list'] = simple_names
                info['drug_count'] = len(simple_names)
                info['drug_details'] = drugs_info

            # Build subgraph (only nodes within cutoff hops from any biomarker)
            H = G.subgraph(subgraph_nodes).copy()

            node_list = []
            for n in H.nodes():
                node_list.append({
                    'id': n,
                    'label': n,
                    'is_biomarker': n in biomarkers,
                    'degree': int(H.degree(n))
                })

            edge_list = []
            for u, v, data in H.edges(data=True):
                edge_list.append({
                    'source': u,
                    'target': v,
                    'weight': float(data.get('weight', 1.0))
                })

            candidates = sorted(
                candidates_map.values(),
                key=lambda x: (x['hops_from_biomarker'], -x['score'], x['target'])
            )

            response_data = {
                'status': 'success',
                'cancer_type': cancer_type,
                'biomarkers': biomarkers,
                'graph': {
                    'num_nodes': G.number_of_nodes(),
                    'num_edges': G.number_of_edges(),
                    'subgraph': {
                        'nodes': node_list,
                        'edges': edge_list
                    }
                },
                'candidates': candidates
            }

            return JsonResponse(response_data)

        finally:
            if os.path.exists(full_path):
                os.remove(full_path)

    except Exception as e:
        logger.error(f"Drug repurposing engine error: {str(e)}")
        return JsonResponse({'error': f'Drug repurposing failed: {str(e)}'}, status=500)
