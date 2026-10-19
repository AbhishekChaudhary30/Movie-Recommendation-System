# Movie Recommendation System

> **Intelligent Movie Discovery & Recommendation Analytics**  
> *Discover movies you'll love through intelligent content-based filtering, similarity scoring, movie metadata analysis, and personalized recommendations.*

---

## 1. Project Overview

The **Movie Recommendation System** is a portfolio-grade machine learning application built with **Python**, **Pandas**, **NumPy**, **Scikit-learn**, **Streamlit**, and **Plotly**. It provides an end-to-end data engineering and recommendation workflow allowing users to upload any standard movie metadata CSV, automatically detect schemas and synonyms, execute automated cleaning with quality auditing, explore titles via a paginated search explorer, analyze trends through 15+ interactive charts, compare titles side-by-side with pairwise cosine similarity scoring, and receive content-based recommendations accompanied by explainable similarity breakdowns.

---

## 2. Problem Statement

Modern movie streaming platforms offer thousands of titles, making manual search overwhelming and leading to decision fatigue. Basic recommendation tutorials frequently hardcode schemas, ignore malformed text/JSON fields, fabricate similarity scores, or generate dense similarity matrices that crash on realistic datasets. 

This project solves these limitations by providing:
- **Dynamic schema mapping** that detects diverse column synonyms (e.g. `title`, `movie_title`, `overview`, `plot`, `actors`, `cast`, `imdb_rating`).
- **Memory-aware $O(N)$ sparse vector similarity calculation** that avoids dense $N \times N$ matrix memory exhaustion.
- **Automated data cleaning**, exact duplicate pruning, duplicate-title disambiguation, and transparent quality auditing.
- **Multi-attribute movie explorer** with search and paginated card browsing.
- **Side-by-side movie comparison** with direct pairwise cosine similarity calculation.
- **Dual analytics modes**: One-click complete automated analysis and deep-dive manual category exploration.
- **Advanced interactive visualization library** featuring 15+ Plotly charts powered by a central Chart Registry.
- **Transparent recommendation rationales** explaining *why* a movie was matched (shared genres, director, cast, keywords, and thematic terms).

---

## 3. Application Sections & Capabilities

The application is structured into 9 dedicated, cohesive sections:

1. **🏠 Home:** Hero presentation, capability overview, real-time dataset and engine status cards, and guided workflow progression.
2. **📂 Dataset & Data Quality:** Upload custom CSV (up to 50MB) or load the curated 50-movie cinema gem dataset; review detected schema mappings with override controls, inspect quality audits, and download cleaned datasets.
3. **🔍 Movie Explorer:** Interactive multi-attribute catalog explorer; search across titles, directors, cast, and narrative keywords; filter by genre, release era, rating, and runtime with paginated 3x3 movie cards.
4. **🎯 Recommendations:** Select any seed title; configure Top-N, minimum similarity threshold, genre/rating/year filters, and custom feature weights; review match scores and multi-factor explanation rationales.
5. **⚖️ Compare Movies:** Side-by-side comparative analysis for 2 to 4 titles with direct pairwise similarity percentage calculation, shared attribute highlights, and exportable comparison matrix.
6. **📊 Analytics Dashboard:** Executive KPIs, evidence-based data insights, and dual analysis modes:
   - **Mode A (Automatic Analysis):** One-click execution of complete analytical pipeline with downloadable Executive Summary Report.
   - **Mode B (Manual Mode):** Deep-dive category analysis across Genre, Rating, Release Year, Runtime, and Data Quality.
7. **📈 Advanced Visualizations:** Centralized library of 15+ interactive Plotly charts across 6 analytical dimensions with filterable categories and chart data downloads.
8. **📥 Reports & Downloads:** Central hub to download cleaned datasets (CSV), quality audit logs (MD), executive analytics reports (MD), recommendations (CSV), and comparison matrices (CSV).
9. **ℹ️ About This Project:** Architectural overview, mathematical formulation of TF-IDF and Cosine similarity, memory governance, and portfolio interview discussion talking points.

---

## 4. Light Theme Editorial UI Design System

Designed with an editorial cinema aesthetic combining streaming aesthetics with modern analytics software:
- **Background:** Warm ivory neutral (`#FBFBFD`)
- **Card Surfaces:** Crisp white with restrained elevation (`#FFFFFF`)
- **Primary Accent:** Cinematic violet (`#6C5CE7`)
- **Secondary Accent:** Soft teal (`#00B894`)
- **Highlight:** Muted amber (`#E17055`)
- **Typography & Text:** Deep charcoal (`#1E272E`) and neutral gray (`#636E72`)
- **Borders & Dividers:** Subtle gray (`#E2E8F0`)

---

## 5. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+ (tested on Python 3.13) |
| **Data Processing & Engineering** | Pandas (v2.3.2), NumPy (v2.3.3), python-dateutil |
| **Machine Learning & Vectorization** | Scikit-learn (v1.7.2: `TfidfVectorizer`, `linear_kernel`), SciPy |
| **Frontend UI** | Streamlit (v1.49.1) |
| **Interactive Visualizations** | Plotly Express & Plotly Graph Objects (v6.3.0) |
| **Testing & Quality Assurance** | Pytest (v8.4.2: 48 automated unit and integration tests) |
| **CI/CD** | GitHub Actions Workflow (`.github/workflows/tests.yml`) |

---

## 6. Recommendation Methodology

The core engine implements **genuine content-based filtering**:
1. **Metadata Soup Construction:** For each movie, attributes (`genres`, `keywords`, `overview`, `cast`, `director`) are normalized and combined into a weighted representation. Configurable weights repeat key attributes proportionally (e.g. Genre weight 2.0x, Keywords 1.5x, Cast 1.2x).
2. **Entity Tokenization:** Names are converted to unified tokens (e.g. `Christopher Nolan` $\rightarrow$ `christopher_nolan`) to prevent cross-matching common first or last names.
3. **TF-IDF Vectorization:** The combined soup is transformed with sublinear term frequency scaling and English stop-word filtering into a sparse feature matrix.
4. **Sparse Cosine Similarity:** Queries compute on-demand similarity vectors against the dataset in $O(N)$ memory:
   $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
5. **Self-Exclusion & Ranking:** The query movie is strictly excluded from its own recommendations. Results are ranked deterministically by similarity score, rating, and vote count.
6. **Explainability Engine:** Identifies shared genres, overlapping actors, shared director, and matching thematic keywords between the seed movie and each recommendation.

---

## 7. Project Structure

```
movie-recommendation-system/
│
├── app.py                      # Main Streamlit application entrypoint (9 sections)
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation
├── .gitignore                  # Git privacy and exclusion rules
│
├── .streamlit/
│   └── config.toml             # Streamlit light theme & server settings
│
├── .github/
│   └── workflows/
│       └── tests.yml           # CI test automation
│
├── data/
│   ├── raw/                    # Uploaded raw files (git-ignored)
│   ├── processed/              # Cleaned export files (git-ignored)
│   └── sample/
│       └── sample_movies.csv   # Curated 50-movie cinema gem dataset
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Constants, synonym dictionary, design tokens
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Multi-encoding CSV loading & file size limits
│   │   ├── schema.py           # Synonym schema detection and user mapping
│   │   ├── cleaning.py         # Automated data cleaning & audit reporting
│   │   └── validation.py       # Structural and integrity validation
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_processing.py  # Entity tokenization and string sanitization
│   │   └── feature_engineering.py # Weighted metadata soup & TF-IDF
│   │
│   ├── recommendation/
│   │   ├── __init__.py
│   │   ├── similarity.py       # Memory-aware cosine similarity
│   │   └── content_based.py    # Recommender class with filters & pairwise scoring
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Summary KPIs and executive insights
│   │   ├── visualizations.py   # 15+ Plotly charts & ChartRegistry
│   │   └── reports.py          # Executive markdown report & comparison matrix
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   └── components.py       # Reusable light-theme UI cards & helpers
│   │
│   └── utils/
│       ├── __init__.py
│       ├── exceptions.py       # Domain-specific custom exceptions
│       └── logging_config.py   # Standardized logger
│
└── tests/
    ├── __init__.py
    ├── test_loader.py          # Loader and encoding tests (6 tests)
    ├── test_validation.py      # Schema detection & validation tests (6 tests)
    ├── test_cleaning.py        # Cleaning, duplicates & JSON parsing tests (7 tests)
    ├── test_similarity.py      # Cosine similarity and memory guard tests (3 tests)
    ├── test_recommendations.py # Recommendation engine and filter tests (10 tests)
    ├── test_analytics.py       # KPI, insights, and report tests (6 tests)
    ├── test_visualizations.py  # Chart registry and Plotly builders tests (6 tests)
    ├── test_comparison.py      # Pairwise similarity & matrix tests (3 tests)
    └── test_e2e_workflow.py    # Complete 4-stage pipeline integration test (1 test)
```

---

## 8. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone or Open the Repository
```bash
git clone <repo-url>
cd movie-recommendation-system
```

### 2. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. How to Run the Application

Start the Streamlit application locally:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 10. Running Automated Tests

Run the complete test suite with verbose output:
```bash
pytest -v
```
All **48 test cases** pass with zero failures:
- CSV multi-encoding loading and file size boundaries.
- Schema detection across synonyms and user-supplied overrides.
- Missing title rejection and validation diagnostics.
- Duplicate record removal and potential duplicate title disambiguation.
- JSON stringified list parsing (`[{"name": "Action"}]`) and pipe delimiter splitting.
- Memory-safe cosine similarity scoring in $[0, 1]$.
- Recommendation self-exclusion, minimum similarity filter, and deterministic ranking.
- Genre, minimum rating, and release year filters.
- Recommendation explanations generation.
- Executive KPI calculations, median ratings, most common release year, and evidence-based insights.
- 15+ Plotly charts and ChartRegistry schema validation.
- Direct pairwise similarity computation and movie comparison matrix.
- Full end-to-end integration workflow.

---

## 11. Security, Privacy & Deployment Preparation

### Data Privacy & Git Policy
- The `.gitignore` is configured to strictly exclude all raw uploaded data (`data/raw/*`) and processed files (`data/processed/*`).
- Only the sanitized sample dataset (`data/sample/sample_movies.csv`) is version-controlled.
- No API keys, credentials, or personal user data are stored.

### Deployment Readiness
The repository is structured for zero-configuration deployment to **Streamlit Community Cloud**:
- Root `app.py` entrypoint.
- Explicit `requirements.txt` with tested version pins.
- Configured `.streamlit/config.toml` enforcing headless mode and responsive light theme.

---

## 12. Known Limitations & Roadmap

### Current Scope
- The recommendation engine is strictly **content-based**. It does not perform collaborative filtering (no user-item interaction matrices or matrix factorization).
- Vectorization relies on TF-IDF word frequencies rather than deep dense neural embeddings (e.g. SBERT).

### Roadmap for Final Stage (Master Prompt 3)
- Hybrid recommendation exploration & simulated collaborative filtering evaluation.
- Model performance evaluation metrics (Precision@K, Recall@K, Diversity, Catalog Coverage).
- Streamlit Community Cloud production deployment verification and final portfolio presentation polish.
