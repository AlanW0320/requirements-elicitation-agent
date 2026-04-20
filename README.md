# AI Requirements Elicitation Agent

An AI-powered agent that classifies software requirements, clarifies vague stakeholder inputs through a dialogue loop, and generates structured Software Requirements Specification (SRS) documents automatically.

Built as an academic project using a fine-tuned BERT classifier combined with GPT-4o for clarification and document generation.

---

## Overview

Requirements elicitation is a critical but time-intensive phase in software development. Analysts must manually read stakeholder inputs, categorise each statement as functional or non-functional, and document them in a structured format. This project automates that process through a three-stage pipeline:

1. **Classify** — A fine-tuned BERT model classifies each input into 13 categories (FR, NFR subcategories, or Ambiguous)
2. **Clarify** — Vague or ambiguous inputs are refined through a GPT-4o dialogue loop that generates targeted clarification questions
3. **Document** — All classified requirements are pulled from a registry and compiled into an IEEE 830 formatted SRS `.docx` file

---

## Features

- **13-class BERT classifier** trained on PROMISE, PURE, and FR_NFR datasets
- **Confidence-based routing** — inputs below the threshold are flagged as ambiguous and sent to clarification
- **GPT-4o clarification loop** — generates targeted questions, refines vague requirements, re-classifies
- **Compound requirement splitting** — detects and separates multiple requirements in a single input
- **Grammar and completeness checking** — fixes typos, missing actors, and vague pronouns before saving
- **Requirements registry** — SQLite-backed persistent store for all classified requirements
- **SRS document generator** — produces IEEE 830 structured `.docx` files from the registry
- **Streamlit UI** — interactive web interface with Interactive and Auto clarification modes

---

## Project Structure

```
requirements-agent/
│
├── app.py                          # Streamlit application entry point
│
├── agent/
│   ├── __init__.py
│   ├── classifier.py               # BERT classifier + GPT second opinion
│   ├── clarification.py            # Ambiguity analysis, question generation, refinement
│   └── srs_generator.py            # IEEE 830 SRS document generation
│
├── models/
│   └── bert_requirements_final/    # Saved fine-tuned BERT model (not tracked by Git)
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       ├── label2id.json
│       └── id2label.json
│
├── data/
│   ├── raw/                        # Original downloaded datasets (not tracked by Git)
│   ├── processed/                  # Cleaned and merged datasets
│   │   ├── master_dataset_balanced.csv
│   │   ├── label2id.json
│   │   └── id2label.json
│   └── requirements_registry.db   # SQLite registry
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_bert_training.ipynb
│   ├── 03_clarification_agent.ipynb
│   └── 04_srs_generator.ipynb
│
├── outputs/                        # Generated SRS documents
│
├── .env.example                    # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML Framework | PyTorch 2.5.1 + HuggingFace Transformers |
| Base Model | bert-base-uncased |
| LLM API | OpenAI GPT-4o / GPT-4o-mini |
| Data Processing | Pandas, NumPy, NLTK, SymSpellPy |
| Application | Streamlit |
| Database | SQLite |
| Document Export | python-docx |
| Version Control | Git + GitHub |

---

## Datasets

The model was trained on three publicly available datasets:

| Dataset | Source | Samples | Labels |
|---|---|---|---|
| PROMISE NFR (expanded) | [Zenodo](https://doi.org/10.5281/zenodo.268542) | 969 | FR + 11 NFR subcategories |
| PURE | [Zenodo](https://doi.org/10.5281/zenodo.1414117) | ~2,800 (auto-labeled) | FR + NFR subcategories |
| FR_NFR | [Mendeley](https://data.mendeley.com/datasets/4ysx9fyzv4/1) | 6,118 | FR / NFR binary |
| Synthetic Ambiguous | GPT-4o-mini generated | 300 | Ambiguous |

> **Note:** Datasets are not included in this repository. Download them from the sources above and place them in `data/raw/`.

---

## Model Performance

Final model trained on 13 classes with the fine-tuned BERT classifier:

| Metric | Score |
|---|---|
| Accuracy | 0.8561 |
| Macro F1 | 0.8400 |
| Macro Precision | 0.8402 |
| Macro Recall | 0.8479 |

**13 Classes:**
`FR` `NFR_Performance` `NFR_Security` `NFR_Usability` `NFR_Reliability` `NFR_Maintainability` `NFR_Scalability` `NFR_Portability` `NFR_Operational` `NFR_Legal` `NFR_LookAndFeel` `NFR_Other` `Ambiguous`

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/AlanW0320/requirements-elicitation-agent.git
cd requirements-elicitation-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Copy the `.env.example` file and add your OpenAI API key:

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```
OPENAI_API_KEY=your-openai-api-key-here
```

> Never commit your `.env` file. It is already listed in `.gitignore`.

### 5. Download and place the trained model

Download the trained BERT model and place it in `models/bert_requirements_final/`. The model is not tracked by Git due to file size. If you want to train it yourself, run the notebooks in order starting from `01_data_preprocessing.ipynb`.

### 6. Run the application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Running the Notebooks

If you want to reproduce the training pipeline from scratch, run the notebooks in this order:

```
01_data_preprocessing.ipynb   — load, clean, merge, and balance all datasets
02_bert_training.ipynb         — fine-tune BERT and evaluate on test set
03_clarification_agent.ipynb   — build and test the clarification loop
04_srs_generator.ipynb         — test the SRS generation pipeline
```

Make sure your datasets are placed in `data/raw/` before running notebook 01.

---

## How It Works

### Agent Loop

```
User submits requirement
        ↓
Split compound requirements (GPT-4o)
        ↓
BERT classifies each requirement
        ↓
    ┌───┴──────────────────┐
    │                      │
Ambiguous              FR or NFR
    │                      │
GPT-4o generates       GPT-4o checks grammar
clarification          and missing actors
questions                  │
    │                  Stores to registry
User answers
    │
GPT-4o refines
requirement
    │
BERT re-classifies
and stores to registry
        ↓
User requests SRS generation
        ↓
SRS generator pulls all requirements from registry
        ↓
Complete IEEE 830 SRS document (.docx)
```

### Confidence Thresholds

| Confidence | Action |
|---|---|
| < 0.55 | Flagged as Ambiguous → clarification loop |
| 0.55 – 0.75 | GPT-4o second opinion → confirm or correct label |
| ≥ 0.75 | BERT label accepted directly |

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key — required for all GPT calls |

---

## Known Limitations

- `NFR_Operational` achieved a lower F1-score (0.74) due to label noise in the auto-labeled PURE dataset. Some sentences describing hardware compatibility and performance monitoring were incorrectly labeled as operational during GPT auto-labeling.
- The auto mode refines vague requirements using inferred context from GPT-4o. The assumption made during refinement is displayed to the user but should be reviewed before finalising the SRS.
- SRS generation quality depends on the number and variety of requirements in the registry. A minimum of 5 classified requirements is recommended before generating a document.

---

## Project Objectives

This project was developed to address three research objectives:

1. Classify functional requirements (FRs), non-functional requirements (NFRs), and ambiguous statements from stakeholder inputs
2. Implement an automated clarification mechanism that generates targeted questions when input is vague
3. Generate a draft SRS document from classified requirements in IEEE 830 format

---

## Academic Context

This project was developed as an academic final year project. The datasets used are publicly available for research purposes. Please refer to each dataset's original license before using them in other projects.

**Datasets used:**
- PROMISE NFR — attributed to Cleland-Huang et al. (2007), available under Creative Commons Attribution license
- PURE — attributed to Ferrari et al. (2017), available on Zenodo
- FR_NFR — available on Mendeley Data (2024)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The datasets used for training are subject to their own respective licenses and are not redistributed in this repository.

---

## Author

**Alan Wong**   
Final Year Project — AI Requirements Elicitation Agent for SMEs

---

*Generated SRS documents are drafts intended to assist requirements engineers. Human review is recommended before finalising any SRS document produced by this system.*
