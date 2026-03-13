# Test the new 3-agent system

import requests
import json

def test_new_agents():
    data = {
        "cancer_type": "breast_cancer",
        "biomarkers": [
            {"gene": "BRCA1", "importance": 0.95},
            {"gene": "TP53", "importance": 0.87},
            {"gene": "PIK3CA", "importance": 0.76},
            {"gene": "ERBB2", "importance": 0.68}
        ]
    }
    
    print("Testing NEW 3 agents...")
    response = requests.post("http://127.0.0.1:8000/classification/api/validate-agents/", 
                           json=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS! Orchestrator result:")
        print(json.dumps(result, indent=2)[:1000] + "...")
        print(f"Agents run: {result.get('orchestrator_metadata', {}).get('agents_run', [])}")
        print(f"Overall: {result.get('overall_status')} ({result.get('overall_confidence')})")
    else:
        print(f"❌ ERROR {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_new_agents()

