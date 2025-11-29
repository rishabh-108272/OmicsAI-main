# 🧬 OmicsAI – SaaS Platform for Explainable Multi-Omics Analysis

OmicsAI is a SaaS-based platform designed to automate and simplify multi-omics analysis for cancer research.  
It provides an end-to-end workflow combining AI-based classification, explainable biomarker discovery, drug repurposing, literature summarization, and 3D structural analysis — all without requiring any coding expertise.

---

## 🚀 Overview

OmicsAI enables clinicians, researchers, and bioinformaticians to analyze single-patient gene expression data using a fully automated 5-step pipeline:

1. **Cancer Subtype Prediction**  
2. **Biomarker Discovery using XAI (SHAP/LIME)**  
3. **Drug Repurposing via Knowledge Graphs**  
4. **AI Research Agent (RAG) for Literature Summaries**  
5. **Protein Structure Prediction with AlphaFold3**

---

## 🧠 Key Features

### 🔬 1. AI-Based Cancer Classification
- Supports Liver, Lung, and Colorectal cancer models  
- Built using Logistic Regression, Random Forest, MLP, and Ensemble Models  
- Provides confidence scores for predictions  

### 🧩 2. XAI-Powered Biomarker Discovery
- SHAP and LIME visualizations  
- Identifies top contributing biomarkers for each prediction  

### 💊 3. Drug Repurposing Engine
- Built using STRING Protein-Protein Interaction Networks  
- Graph traversal (BFS) suggests drugs targeting biomarkers or their key neighbors  

### 📚 4. AI Research Agent (RAG)
- Retrieves scientific literature via custom embeddings  
- Generates concise summaries using LLMs from HuggingFace  

### 🧫 5. Structural Analysis with AlphaFold3
- Predicts 3D protein structures  
- Interactive molecular viewer for visualization  

---

## 🏗️ System Architecture

### **Frontend**
- React.js  
- Tailwind CSS  
- Interactive dashboards & visualization panels  

### **Backend**
- Django + FastAPI (planned microservices architecture)  
- Python ML libraries (Scikit-Learn, TensorFlow, SHAP, NetworkX)  
- REST API integration with HuggingFace & AlphaFold3  
- JWT authentication (planned)

### **Database**
- PostgreSQL / MongoDB (planned)  
- Stores user sessions, analysis outputs, and processing logs  

---

## 📁 Workflow Pipeline


All results are displayed on a unified dashboard for rapid interpretation.

---

## 📊 Machine Learning Models

### **Liver Cancer (Ensemble Model)**
- Random Forest, Logistic Regression, Naive Bayes  
- Accuracy: **99.22%**

### **Colorectal Cancer**
- Logistic Regression (best model)  
- Accuracy: **94.44%**

### **Lung Cancer**
- Multi-Layer Perceptron (RNA-Seq)  
- Multi-omics models: LR, XGBoost, RF  

---

## 📦 Tech Stack

**Frontend:** React.js, Tailwind CSS  
**Backend:** Django, FastAPI, Python  
**ML/XAI:** Scikit-Learn, TensorFlow, SHAP, LIME  
**Knowledge Graph:** NetworkX, STRING DB  
**APIs:** HuggingFace, AlphaFold3  
**Database:** PostgreSQL / MongoDB (planned)

---

## 🛠️ Installation

### Backend Setup
```bash
git clone [https://github.com/rishabh-108272/OmicsAI.git](https://github.com/rishabh-108272/OmicsAI-main.git)
cd OmicsAI
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver

cd frontend
npm install
npm run dev
