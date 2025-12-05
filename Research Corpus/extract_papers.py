import os 
import json 
import re 
from typing import List, Dict, Tuple, Optional 

import fitz 
from tqdm import tqdm 

INPUT_DIR="./papers"
OUTPUT_JSON="data/papers.json"

os.makedirs(os.path.dirname(OUTPUT_JSON),exist_ok=True)

KNOWN_GENES = {
    "FASN", "ACACA", "EGFR", "ALK", "KRAS", "NRAS", "HRAS",
    "BRAF", "TP53", "ATM", "CHEK1", "CHEK2", "PIK3CA",
    "PTEN", "BRCA1", "BRCA2", "APC", "CTNNB1", "MLH1", "MSH2",
}

KNOWN_DRUGS = {
    # NSCLC targeted drugs
    "Gefitinib", "Erlotinib", "Osimertinib", "Afatinib", "Dacomitinib",
    "Alectinib", "Crizotinib", "Ceritinib", "Brigatinib", "Lorlatinib",
    "TVB-2640", "Amuvatinib",

    # CRC-related drugs
    "Cetuximab", "Panitumumab", "Bevacizumab", "Regorafenib",
    "Trifluridine", "Tipiracil",

    # Immunotherapy (common across)
    "Pembrolizumab", "Nivolumab", "Atezolizumab", "Durvalumab",
}

def extract_text_from_pdf(path: str)-> str:
    doc=fitz.open(path)
    texts=[]
    for page in doc:
        texts.append(page.get_text())
    doc.close()
    return "\n".join(texts)

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def guess_title_and_abstract(text: str)-> Tuple[str, str]:
    lines=[ln.strip() for ln in text.splitlines()]
    lines=[ln for ln in lines if ln]
    
    title=lines[0] if lines else "Unknown Title"
    
    abstract=""
    joined="\n".join(lines)
    
    abstract_match=re.search(
        r"(Abstract|ABSTRACT|Summary|SUMMARY)\s*[:\n](.+?)(\n[A-Z][A-Za-z]{3,20}\n|\Z)",
        joined,
        flags=re.DOTALL,
    )
    
    if abstract_match:
        abstract=abstract_match.group(2).strip()
    else:
        abstract=joined[:1500].strip()
    
    return title, abstract 


def find_terms(text:str, terms: List[str]) -> List[str]:
    
    found=set()
    lower_text=text.lower() 
    
    for term in terms:
        term_str=str(term)
        if not term_str:
            continue 
        
        pattern=r"\b" + re.escape(term_str)+ r"\b"
        
        if re.search(pattern, lower_text, flags=re.IGNORECASE):
            found.add(term_str)
    return sorted(found)


def guess_cancer_type(text: str) -> str:
    """
    Very simple heuristic classifier for NSCLC vs CRC.
    """
    t = text.lower()

    nsclc_keywords = [
        "non-small cell lung cancer", "nsclc",
        "lung adenocarcinoma", "luad",
        "lung squamous cell carcinoma", "lusc",
        "lung cancer"
    ]
    crc_keywords = [
        "colorectal cancer", "crc",
        "colon cancer", "rectal cancer"
    ]

    nsclc_score = sum(1 for kw in nsclc_keywords if kw in t)
    crc_score = sum(1 for kw in crc_keywords if kw in t)

    if nsclc_score > crc_score and nsclc_score > 0:
        return "NSCLC"
    if crc_score > nsclc_score and crc_score > 0:
        return "CRC"
    if nsclc_score == 0 and crc_score == 0:
        return "UNKNOWN"
    # tie-breaker
    return "MIXED"

def process_paper(path: str, idx: int) -> Dict:
    """
    Process a single PDF/TXT file and return a structured record.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(path)
    else:
        raw_text = extract_text_from_txt(path)

    title, abstract = guess_title_and_abstract(raw_text)

    genes = find_terms(raw_text, list(KNOWN_GENES))
    drugs = find_terms(raw_text, list(KNOWN_DRUGS))
    cancer_type = guess_cancer_type(raw_text)

    record = {
        "id": f"paper_{idx:03d}",
        "file_name": os.path.basename(path),
        "title": title,
        "cancer_type": cancer_type,
        "genes": genes,
        "drugs": drugs,
        "text": abstract,  # for RAG, we mainly use the abstract/summary
    }
    return record


def main():
    files = []
    for name in os.listdir(INPUT_DIR):
        if name.lower().endswith(".pdf") or name.lower().endswith(".txt"):
            files.append(os.path.join(INPUT_DIR, name))

    files = sorted(files)
    print(f"Found {len(files)} files in {INPUT_DIR}")

    records = []
    for i, path in enumerate(tqdm(files, desc="Processing papers")):
        try:
            rec = process_paper(path, i + 1)
            records.append(rec)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(records)} records to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
    