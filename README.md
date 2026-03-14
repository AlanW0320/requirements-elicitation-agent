# AI Requirements Elicitation Agent

An AI-powered agent for classifying functional requirements (FR), 
non-functional requirements (NFR), and ambiguous statements from 
stakeholder inputs, with automated clarification and SRS generation.

## Project Objectives
- Classify FRs, NFRs, and ambiguous statements using fine-tuned BERT
- Implement automated clarification for vague requirements using LLMs
- Generate draft SRS documents from classified requirements

## Datasets Used
- PROMISE NFR Dataset
- PURE Dataset  
- FR_NFR Dataset (Mendeley)

## Tech Stack
- Python 3.10+
- HuggingFace Transformers (BERT)
- LangChain
- FastAPI
- Streamlit

## Project Structure
```
requirements-agent/
├── data/processed/     # Label maps (datasets kept local)
├── notebooks/          # Jupyter notebooks (step by step)
├── agent/              # Agent orchestration code
├── models/             # Saved model checkpoints (local only)
└── outputs/            # Generated SRS documents
```

## Setup
```bash
pip install -r requirements.txt
```

## Status
🔄 In Progress — Academic Project
