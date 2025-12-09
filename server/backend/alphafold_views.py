# backend/alphafold_views.py

import logging
import requests
from django.http import JsonResponse
from django.views.decorators.http import require_GET

logger = logging.getLogger(__name__)

ALPHAFOLD_BASE_URL = "https://alphafold.ebi.ac.uk/api/prediction"


@require_GET
def alphafold_prediction(request, accession: str):
    """
    Proxy to AlphaFold EBI API:
    GET /api/alphafold/prediction/<accession>/?sequence_checksum=...
    
    - accession: UniProt accession (e.g. Q5VSL9)
    - sequence_checksum: optional MD5 checksum of the UniProt sequence
    """

    sequence_checksum = request.GET.get("sequence_checksum")

    # Build EBI URL: /prediction/{qualifier}
    url = f"{ALPHAFOLD_BASE_URL}/{accession}"

    # Forward query params only if present
    params = {}
    if sequence_checksum:
        params["sequence_checksum"] = sequence_checksum

    try:
        resp = requests.get(url, params=params, timeout=15)
    except requests.RequestException as e:
        logger.exception("Error calling AlphaFold EBI API")
        return JsonResponse(
            {
                "error": "Failed to contact AlphaFold EBI API.",
                "details": str(e),
            },
            status=502,
        )

    # If EBI returns non-200, forward the status & some info
    if resp.status_code != 200:
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw_text": resp.text}

        return JsonResponse(
            {
                "error": "AlphaFold EBI API returned an error.",
                "status_code": resp.status_code,
                "ebi_response": payload,
            },
            status=resp.status_code,
        )

    # Normal success: return their JSON directly
    try:
        data = resp.json()
    except Exception:
        # If they somehow didn't return JSON
        return JsonResponse(
            {
                "error": "AlphaFold EBI API did not return valid JSON.",
                "raw_text": resp.text[:5000],
            },
            status=502,
        )

    # Optionally you could post-process here (e.g., extract pdbUrl only)
    return JsonResponse(data, safe=False)  # data is a list
