# Movie Recommendation System

> **Intelligent Movie Discovery & Recommendation Analytics**  
> *Discover movies you'll love through intelligent content-based filtering, similarity scoring, movie metadata analysis, and personalized recommendations.*

---

## 1. Project Overview

The **Movie Recommendation System** is a portfolio-grade machine learning application built with **Python**, **Pandas**, **NumPy**, **Scikit-learn**, and **Streamlit**. It provides an end-to-end data engineering and recommendation workflow that allows users to upload any standard movie metadata CSV, automatically detect schemas and synonyms, execute automated cleaning with quality auditing, and receive content-based recommendations accompanied by explainable similarity breakdowns.

---

## 2. Problem Statement

Modern movie streaming platforms offer thousands of titles, making manual search overwhelming and leading to decision fatigue. Basic recommendation tutorials frequently hardcode schemas, ignore malformed text/JSON fields, fabricate similarity scores, or generate dense similarity matrices that crash on realistic datasets. 

This project solves these limitations by providing:
- Dynamic schema mapping that detects diverse column synonyms (e.g. `title`, `movie_title`, `overview`, `plot`, `actors`, `cast`, `imdb_rating`).
- Memory-aware $O(N)$ sparse vector similarity calculation that avoids dense $N \times N$ matrix memory exhaustion.
- Automated data cleaning, exact duplicate pruning, duplicate-title disambiguation, and transparent quality auditing.
- Transparent recommendation rationales explaining *why* a movie was matched (shared genres, director, cast, keywords, and thematic terms).

---

## 3. Features & Architecture Highlights

### Step-by-Step 4-Stage Workflow with Progression Enforcers
1. **Stage 1 — Dataset Selection & Ingestion:** Choose from the built-in curated 50-movie cinema gem dataset or upload a custom movie CSV (up to 50MB) with automatic encoding detection (`utf-8`, `latin-1`, `cp1252`).
2. **Stage 2 — Schema Mapping & Validation:** Automatic recognition of canonical columns, user mapping overrides, and critical validation ensuring movie identification fields exist.
3. **Stage 3 — Data Cleaning & Quality Audit:** Automatic elimination of blank rows/columns, exact duplicate removal, nested JSON and delimited list parsing, negative value sanitization, and downloadable audit reports (Markdown/CSV).
4. **Stage 4 — Interactive Recommendation Engine & Analytics:** Seed movie selection, configurable Top-N, dynamic genre filter, minimum rating slider, release year bounds, and customizable feature weights.

### Light Theme Editorial UI
Designed with an editorial cinema aesthetic:
- **Background:** Warm ivory neutral (`#FBFBFD`)
- **Card Surfaces:** Crisp white with restrained elevation (`#FFFFFF`)
- **Primary Accent:** Cinematic violet (`#6C5CE7`)
- **Secondary Accent:** Soft teal (`#00B894`)
- **Highlight:** Muted amber (`#E17055`)
- **Typography & Text:** Deep charcoal (`#1E272E`) and neutral gray (`#636E72`)

---

## 4. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+ (tested on Python 3.13) |
| **Data Processing & Engineering** | Pandas, NumPy, python-dateutil |
| **Machine Learning & Vectorization** | Scikit-learn (`TfidfVectorizer`, `linear_kernel`), SciPy |
| **Frontend UI** | Streamlit |
| **Interactive Visualizations** | Plotly Express & Plotly Graph Objects |
| **Testing & Quality Assurance** | Pytest (32 automated unit and integration tests) |
| **CI/CD** | GitHub Actions Workflow (`.github/workflows/tests.yml`) |

---

## 5. Recommendation Methodology

The core engine implements **genuine content-based filtering**:
1. **Metadata Soup Construction:** For each movie, attributes (`genres`, `keywords`, `overview`, `cast`, `director`) are normalized and combined into a weighted representation. Configurable weights repeat key attributes proportionally (e.g. Genre weight 2.0x, Keywords 1.5x, Cast 1.2x).
2. **Entity Tokenization:** Names are converted to unified tokens (e.g. `Christopher Nolan` $\rightarrow$ `christopher_nolan`) to prevent cross-matching common first or last names.
3. **TF-IDF Vectorization:** The combined soup is transformed with sublinear term frequency scaling and English stop-word filtering into a sparse feature matrix.
4. **Sparse Cosine Similarity:** Queries compute on-demand similarity vectors against the dataset in $O(N)$ memory:
   $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$
5. **Self-Exclusion & Ranking:** The query movie is strictly excluded from its own recommendations. Results are ranked deterministically by similarity score, rating, and vote count.
6. **Explainability Engine:** Identifies shared genres, overlapping actors, shared director, and matching thematic keywords between the seed movie and each recommendation.

---

## 6. Project Structure

```
movie-recommendation-system/
│
├── app.py                      # Main Streamlit application entrypoint
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
│   │   ├── loader.py           # Robust CSV loading and encoding detection
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
│   │   └── content_based.py    # Recommender class with filters & explanations
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   └── metrics.py          # Summary metrics and Plotly charts
│   │
│   └── utils/
│       ├── __init__.py
│       ├── exceptions.py       # Domain-specific custom exceptions
│       └── logging_config.py   # Configured logger
│
└── tests/
    ├── __init__.py
    ├── test_loader.py          # Loader and encoding tests
    ├── test_validation.py      # Schema detection & validation tests
    ├── test_cleaning.py        # Cleaning, duplicates & JSON parsing tests
    ├── test_similarity.py      # Cosine similarity and memory guard tests
    ├── test_recommendations.py # Recommendation engine and filter tests
    └── test_e2e_workflow.py    # Complete 4-stage pipeline integration test
```

---

## 7. Installation & Setup

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

## 8. How to Run the Application

Start the Streamlit application locally:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 9. Running Automated Tests

Run the complete test suite with verbose output:
```bash
pytest -v
```
All 32 test cases cover:
- CSV multi-encoding loading and file size boundaries.
- Schema detection across synonyms and user-supplied overrides.
- Missing title rejection and validation diagnostics.
- Duplicate record removal and potential duplicate title disambiguation.
- JSON stringified list parsing (`[{"name": "Action"}]`) and pipe delimiter splitting.
- Memory-safe cosine similarity scoring in $[0, 1]$.
- Recommendation self-exclusion and deterministic ranking.
- Genre, minimum rating, and release year filters.
- Recommendation explanations generation.
- Full end-to-end integration workflow.

---

## 10. Dataset Requirements & Compatibility

### Supported Inputs
- Any CSV format with at least a **movie title** column.
- Automatically handles varied column headers:
  - **Title:** `title`, `movie_title`, `original_title`, `film_title`, `name`
  - **Genres:** `genres`, `genre`, `categories`
  - **Overview:** `overview`, `description`, `plot`, `summary`, `storyline`
  - **Keywords:** `keywords`, `tags`, `plot_keywords`
  - **Cast:** `cast`, `actors`, `starring`, `stars`
  - **Director:** `director`, `directors`, `crew`
  - **Release Date:** `release_date`, `release_year`, `year`, `released`
  - **Rating:** `vote_average`, `rating`, `imdb_rating`, `score`
  - **Runtime:** `runtime`, `duration`, `length`

### Unsupported Inputs / Limitations
- Non-CSV files (Excel `.xlsx` or SQLite) are not currently accepted in Stage 1.
- Files exceeding 50MB (to protect cloud deployment resources).
- Files with zero movie identification/title fields.

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

### Current Stage 1 Scope
- The recommendation engine is strictly **content-based**. It does not perform collaborative filtering (no user-item interaction matrices or matrix factorization).
- Vectorization relies on TF-IDF word frequencies rather than deep dense neural embeddings (e.g. SBERT).

### Roadmap for Stages 2 & 3
- Advanced multi-criteria search and hybrid filtering controls.
- Interactive multi-dimensional visualization workbench.
- Poster image integration and external metadata enrichment.
- Model performance evaluation metrics (Precision@K, Coverage, Diversity).

---

## 13. Visual Interface Preview

| Step 1: Ingestion | Step 2: Schema Validation |
| :---: | :---: |
| *Curated 50-movie sample or custom CSV upload* | *Automatic synonym detection & mapping overrides* |

| Step 3: Cleaning & Audit | Step 4: Recommendations & Rationale |
| :---: | :---: |
| *Deduplication, normalization & exportable audit reports* | *Match strength badges, explanations & Plotly distributions* |
