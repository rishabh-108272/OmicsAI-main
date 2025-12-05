import json
import os

INPUT_JSON = "data/papers.json"
OUTPUT_JSON = "data/papers_biomarker.json"

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records")

    filtered = []
    for rec in records:
        cancer_type = rec.get("cancer_type", "UNKNOWN")
        genes = rec.get("genes", [])
        drugs = rec.get("drugs", [])

        # Keep only NSCLC or CRC AND at least one gene or drug
        if cancer_type in ("NSCLC", "CRC") and (genes or drugs):
            filtered.append(rec)

    print(f"Keeping {len(filtered)} biomarker/drug-focused cancer papers")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"Saved filtered corpus to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
