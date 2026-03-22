import json
import logging
import requests

logger = logging.getLogger(__name__)

# Utility for agents - make importable
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

