# Movie Recommendation System

### CineScope — Intelligent Movie Discovery & Recommendation Analytics

> *"CineScope transforms movie metadata into meaningful insights and relevant movie recommendations through content-based machine learning, interactive analytics, and explainable similarity scoring."*

---

## 1. Project Title & Identity

- **Formal Project Name:** Movie Recommendation System
- **Application Brand:** **CineScope**
- **Subtitle:** Intelligent Movie Discovery & Recommendation Analytics
- **Version:** `1.0.0`
- **Domain:** Content-Based Recommendation Systems, Data Engineering, and Exploratory Data Analytics (EDA)

---

## 2. Project Overview

**CineScope** is a production-quality, portfolio-ready Machine Learning and Data Engineering application developed using Python, Scikit-learn, Pandas, NumPy, Streamlit, and Plotly. 

The application delivers an end-to-end analytical user journey:
1. Ingests raw movie metadata CSV files with automatic multi-encoding support (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`).
2. Detects diverse column synonyms across 16 canonical schema fields.
3. Performs automated data cleaning, exact duplicate row pruning, remake title disambiguation, and transparent quality auditing.
4. Explores the movie catalog through a responsive search and paginated card browser.
5. Computes personalized content-based movie recommendations using sublinear TF-IDF vector space modeling and memory-aware sparse cosine similarity.
6. Compares multiple titles side-by-side with on-the-fly direct pairwise similarity scoring.
7. Uncovers catalog patterns across 15+ interactive Plotly visualizations registered in a modular Chart Registry.
8. Generates and exports executive analytics summaries, quality audit logs, and recommendation matrices.

---

## 3. Problem Statement

Modern cinema streaming platforms and catalog databases offer tens of thousands of movies. Discovering relevant films is frequently hindered by:
- **Decision Fatigue**: Users spend more time browsing than watching.
- **Tutorial Oversimplification**: Typical machine learning tutorials rely on rigid CSV columns, fabricate synthetic ratings, fail on malformed strings or nested TMDB JSON structures, and build dense $N \times N$ similarity matrices that cause fatal out-of-memory crashes on realistic catalogs.
- **Opaque Black-Box Matching**: Users are presented with recommendations without understanding *why* a particular movie was recommended.

CineScope solves these challenges by providing:
- **Dynamic Schema Synonym Detection**: Tolerates arbitrary column names (`movie_title`, `original_title`, `actors`, `cast`, `imdb_rating`, etc.).
- **Defensive Data Engineering**: Cleans malformed dates, parses nested JSON / delimited strings, sanitizes corrupted numbers, and logs an auditable quality report.
- **Memory-Aware $O(N)$ Cosine Similarity**: Evaluates query vectors against sparse compressed feature matrices, maintaining instantaneous execution ($<5$ milliseconds) in minimal RAM.
- **Explainable Recommendation Attribution**: Details exact shared genres, matching director, overlapping cast members, and top thematic keyword terms.
- **Non-Commercial Boundaries**: Transparently communicates that recommendations derive from descriptive catalog metadata rather than private streaming histories.

---

## 4. Key Features & Capabilities

| Capability | Engineering Implementation |
| :--- | :--- |
| **Multi-Encoding Ingestion** | Reads UTF-8, Latin-1, and Windows CP1252 encodings; validates file sizes up to 50MB. |
| **Automated Schema Mapping** | Multi-synonym pattern matching across 16 canonical attributes with user override capability. |
| **Data Quality & Audit Pipeline** | Prunes exact duplicates, disambiguates identical titles with release years, sanitizes negative values, and produces Markdown/JSON audit logs. |
| **Interactive Movie Explorer** | Real-time text search across titles, directors, cast, and summaries; multi-attribute filters (genre, release era, rating, runtime); paginated 3x3 card browsing. |
| **Content-Based ML Engine** | Entity tokenization, weighted metadata soup creation, sublinear TF-IDF scaling, and self-exclusion guarantee. |
| **Explainable Matching** | Multi-factor rationale explaining shared genres, cast members, director match, and overlapping TF-IDF thematic terms. |
| **Side-by-Side Comparison** | Multi-movie comparative matrix with on-the-fly direct pairwise cosine similarity calculation. |
| **Dual Analytics Engine** | One-click Automated Complete Analysis Pipeline (with 6 data-driven executive findings) and Manual Category Exploration. |
| **15+ Plotly Visualizations** | Central Chart Registry powering genre distributions, co-occurrence matrices, rating quartiles, release year trends, and completeness charts. |
| **Comprehensive Reporting** | One-click downloads for cleaned datasets (CSV), quality audit logs (MD), recommendations (CSV), comparison matrices (CSV), and executive reports (MD). |

---

## 5. Architectural Overview

The application follows a strictly decoupled, modular architecture where data processing, machine learning logic, and visual presentation are cleanly isolated.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CineScope Streamlit App                           │
│   app.py (9 Integrated Sections • Cinematic Neutral Design System)          │
└───────────────────────▲─────────────────────────────▲───────────────────────┘
                        │                             │
        ┌───────────────┴─────────────┐ ┌─────────────┴───────────────┐
        │   src/analytics/            │ │   src/recommendation/       │
        │   • metrics.py (KPIs)       │ │   • content_based.py        │
        │   • visualizations.py (15+) │ │   • similarity.py           │
        │   • reports.py (Markdown)   │ │   • explanations            │
        └───────────────▲─────────────┘ └─────────────▲───────────────┘
                        │                             │
        ┌───────────────┴─────────────────────────────┴───────────────┐
        │                 src/preprocessing/                          │
        │   • text_processing.py (Entity Tokenizer e.g. tom_hanks)    │
        │   • feature_engineering.py (Weighted Metadata Soup & TF-IDF)│
        └──────────────────────────────▲──────────────────────────────┘
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        │                 src/data/                                   │
        │   • loader.py (Multi-encoding CSV reader & 50MB limit)      │
        │   • schema.py (16 canonical fields & synonym detection)     │
        │   • validation.py (Structural validation & diagnostics)     │
        │   • cleaning.py (Deduplication, title disambiguation)       │
        └─────────────────────────────────────────────────────────────┘
```

- **Separation of Concerns**: Core data, recommendation, and analytics modules contain pure Python logic completely decoupled from Streamlit UI elements, allowing 100% unit testing coverage in headless CI environments.
- **State Machine Safety**: Dynamic dataset transitions instantly flush stale similarity matrices, cached recommendations, and active filters to prevent dirty-read anomalies.

---

## 6. Distinctive Visual Identity: Cinematic Neutral

CineScope implements a refined **Cinematic Neutral** aesthetic inspired by film editorial design and modern analytical products. It avoids both generic all-white SaaS templates and high-contrast dark streaming clones:

| Token Name | Hex Code | Visual Role |
| :--- | :--- | :--- |
| **Main Background** | `#E8E3DA` | Warm stone neutral environment |
| **Main Surface** | `#F5F1E9` | Soft parchment cards and metric containers |
| **Secondary Surface** | `#D5D8CE` | Muted sage-gray badges and tags |
| **Primary Text** | `#252B2B` | Deep ink for high-contrast legibility |
| **Secondary Text** | `#656B68` | Muted slate for captions and descriptions |
| **Primary Accent** | `#76566E` | Restrained cinematic plum for key actions and active states |
| **Secondary Accent**| `#477E79` | Muted petrol teal for status indicators and positive metrics |
| **Highlight** | `#C39A5B` | Subdued ochre for ratings and important callouts |
| **Borders** | `#CDC8BE` | Warm gray boundaries and dividers |

---

## 7. Technology Stack & Dependencies

| Layer | Technologies & Libraries | Verified Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Runtime** | Python | 3.10 – 3.13 | Core programming language |
| **Data Processing** | Pandas | `>= 2.0.0` (2.3.2) | DataFrame manipulation and schema mapping |
| **Numerical Math** | NumPy | `>= 1.24.0` (2.3.3) | Vector operations and array mathematics |
| **Machine Learning**| Scikit-learn | `>= 1.3.0` (1.7.2) | `TfidfVectorizer` and sparse `linear_kernel` |
| **Sparse Algebra** | SciPy | `>= 1.11.0` (1.15.2) | Compressed Sparse Row (`csr_matrix`) structures |
| **User Interface** | Streamlit | `>= 1.30.0` (1.49.1) | Interactive reactive web application shell |
| **Visualizations** | Plotly | `>= 5.18.0` (6.3.0) | Interactive responsive charts |
| **Quality Testing** | Pytest | `>= 7.4.0` (8.4.2) | Automated unit and integration test suite |
| **CI Automation** | GitHub Actions | Ubuntu Latest | Automated testing on push and pull requests |

---

## 8. Recommendation Methodology & Mathematical Foundations

The core engine implements **genuine content-based filtering** powered by vector space modeling:

### 1. Weighted Metadata Soup
For each movie, metadata attributes are combined into a consolidated text document. Configurable weights repeat key terms proportionally to emphasize dominant attributes:
$$\text{Metadata Soup} = \left(\text{Genres} \times w_{\text{genre}}\right) + \left(\text{Keywords} \times w_{\text{kw}}\right) + \left(\text{Cast} \times w_{\text{cast}}\right) + \left(\text{Director} \times w_{\text{dir}}\right) + \left(\text{Overview} \times w_{\text{ov}}\right)$$
*Default Weights:* Genres: 2.0x, Keywords: 1.5x, Cast: 1.2x, Director: 1.0x, Overview: 1.0x.

### 2. Entity Tokenization
Names and multi-word terms are concatenated into single tokens (e.g. `Christopher Nolan` $\rightarrow$ `christopher_nolan`, `Tom Hanks` $\rightarrow$ `tom_hanks`). This prevents spurious unigram overlaps between different individuals sharing first or last names (e.g. *Tom Hanks* vs *Tom Hardy*).

### 3. Sublinear TF-IDF Vector Space Representation
The term frequency-inverse document frequency transformation applies sublinear scaling to prevent high-frequency words from dominating matches:
$$\text{tf-idf}(t, d, D) = \left(1 + \log(\text{tf}(t, d))\right) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
Features are limited to a maximum of 10,000 unigrams and bigrams, filtered for English stopwords.

### 4. Memory-Aware Sparse Cosine Similarity
Rather than allocating a dense $N \times N$ similarity matrix (which requires $O(N^2)$ memory and triggers memory exhaustion on modest catalogs), CineScope computes query similarity on-demand:
$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
Because the matrix is stored in Scipy `csr_matrix` format, computing similarity for movie $i$ involves a single sparse matrix-vector dot product in $O(N)$ memory taking less than 5 milliseconds for 5,000+ movies.

### 5. Multi-Factor Rationale Attribution
Every recommendation isolates verifiable metadata evidence:
- **Shared Genres**: Exact genre intersections (e.g., *Sci-Fi, Adventure*).
- **Shared Director**: Matching director names.
- **Shared Cast**: Overlapping leading actors.
- **Shared Themes**: Top overlapping non-zero TF-IDF vocabulary tokens.

---

## 9. Data Engineering Pipeline

```
Raw CSV Upload (up to 50MB)
  │
  ▼
Encoding Detection (UTF-8, Latin-1, CP1252)
  │
  ▼
Canonical Schema Synonym Detection (16 Fields)
  │
  ▼
Validation Diagnostics (Mandatory Title Check, Type Bounds)
  │
  ▼
Automated Cleaning Pipeline
  ├── Drop entirely empty rows & columns
  ├── Prune exact duplicate records
  ├── Clean and parse delimited/JSON genres, cast, keywords
  ├── Disambiguate duplicate movie titles with release years (e.g. "Avatar (2009)")
  └── Sanitize numeric fields (ratings 0-10, positive runtimes, vote counts)
  │
  ▼
Cleaned Dataset & Quality Audit Report Generated
```

---

## 10. Analytics & Visualization Capabilities

### Executive Metrics & Descriptive Analytics
- **Total Movies & Completeness**: Catalog size and attribute completeness percentage.
- **Rating Summary**: Mean and median audience ratings, interquartile ranges, and outlier identification.
- **Historical Era Coverage**: Earliest and latest release years, primary release decades, and catalog volume trends.
- **Automated Evidence-Based Insights**: Generates 6 narrative findings covering genre leadership, rating health, and runtime clustering.

### Plotly Chart Registry (15+ Interactive Visualizations)
1. **Top Movie Genres**: Horizontal bar chart of most frequent genres.
2. **Genre Share Donut**: Proportional breakdown of leading film categories.
3. **Genre Proportions Treemap**: Hierarchical area mapping of catalog representation.
4. **Genre Co-Occurrence Matrix**: Interactive heatmap of genres paired in the same film.
5. **Rating Distribution & Mean**: Histogram with average benchmark line.
6. **Rating Box Plot & Quartiles**: Quartile summary and outlier detection.
7. **Average Rating by Major Genre**: Comparative quality assessment across genres.
8. **Rating vs. Vote Count Engagement**: Scatter plot evaluating audience engagement vs. rating.
9. **Release Year Distribution**: Chronological distribution of film production.
10. **Movies Produced by Decade**: Volume aggregations across historical film eras.
11. **Average Rating Trend Across Decades**: Historical trendline of film ratings.
12. **Runtime Distribution**: Histogram showing catalog duration and median marker.
13. **Runtime vs. Rating Correlation**: Scatter plot exploring length vs. rating.
14. **Dataset Attribute Completeness**: Horizontal bar chart of metadata health.
15. **Overview Word Count Distribution**: Histogram of narrative synopsis lengths.
16. **Recommendation Match Strength**: Horizontal bar chart of recommendation similarity scores.

---

## 11. Dataset Requirements & Schema Compatibility

CineScope dynamically maps varied naming conventions to 16 canonical attributes:

| Canonical Field | Type | Supported Synonyms | Role in Engine |
| :--- | :--- | :--- | :--- |
| `title` **(Required)** | String | `title`, `movie_title`, `original_title`, `film_title`, `name` | Primary movie identifier |
| `genres` | Text/JSON | `genres`, `genre`, `categories`, `genre_names` | Recommendation feature & filter |
| `overview` | String | `overview`, `description`, `plot`, `summary`, `synopsis` | TF-IDF narrative feature |
| `keywords` | Text/JSON | `keywords`, `keyword`, `tags`, `plot_keywords` | Thematic similarity feature |
| `cast` | Text/JSON | `cast`, `actors`, `starring`, `top_cast`, `lead_actors` | Actor match feature |
| `director` | String | `director`, `directors`, `directed_by`, `crew` | Director match feature |
| `release_date` | Date/Year | `release_date`, `release_year`, `year`, `date`, `released` | Temporal filter & trend analysis |
| `rating` | Float | `vote_average`, `rating`, `imdb_rating`, `score` | Quality filtering & analytics |
| `vote_count` | Integer | `vote_count`, `num_votes`, `votes`, `rating_count` | Engagement analytics & tie-breaking |
| `runtime` | Numeric | `runtime`, `duration`, `length`, `duration_min` | Duration filter & analytics |
| `popularity` | Float | `popularity`, `pop_score`, `trending_score` | Catalog analytics |
| `poster_path` | URL/Path | `poster_path`, `poster_url`, `poster`, `image_url` | Visual rendering |

---

## 12. Installation & Virtual Environment Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Git.

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux (Bash):**
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

## 13. Application Startup Command

To launch CineScope locally:
```bash
streamlit run app.py
```
Open your web browser and navigate to:
```
http://localhost:8501
```

---

## 14. Testing Command & Test Strategy

Execute the full automated test suite with verbose output:
```bash
pytest -v
```

### Test Coverage Summary: 58 Tests Across 9 Suites
- **`tests/test_loader.py`** (6 tests): Valid CSV loading, empty file rejection, zero-row files, Latin-1 encoding, StringIO buffers, file size boundary enforcement.
- **`tests/test_validation.py`** (6 tests): Column name normalization, canonical schema detection, alternate synonym mapping, user overrides, missing title rejection, negative rating diagnostics.
- **`tests/test_cleaning.py`** (7 tests): JSON/pipe delimiter parsing, release year extraction, exact duplicate pruning, duplicate title disambiguation, empty title exclusion, negative numeric sanitization, report serialization.
- **`tests/test_similarity.py`** (3 tests): Query similarity bounds ($[0, 1]$), index bounds checking, full matrix generation.
- **`tests/test_recommendations.py`** (10 tests): Self-exclusion verification, deterministic ranking, genre filters, rating filters, release year filters, similarity thresholding, explanation generation, unindexed movie rejection, missing metadata fallbacks.
- **`tests/test_analytics.py`** (6 tests): KPI calculations, narrative insights generation, decade counts, metadata completeness, executive summary markdown generation, empty catalog handling.
- **`tests/test_visualizations.py`** (6 tests): ChartRegistry availability checks, genre chart builders, rating charts, temporal charts, metadata charts, missing column safety.
- **`tests/test_comparison.py`** (3 tests): Direct pairwise similarity calculation, self-movie similarity ($100\%$), comparison matrix generation.
- **`tests/test_e2e_workflow.py`** (1 test): Complete end-to-end integration test (Upload $\rightarrow$ Validate $\rightarrow$ Clean $\rightarrow$ Recommend $\rightarrow$ Analytics).
- **`tests/test_edge_cases.py`** (10 tests): Single-movie catalog handling, all-optional-columns null, extreme and dirty numeric strings, foreign/Unicode character titles, boundary feature weights, empty filtered results, unindexed movie exception handling, missing title validation rejection, file size limit rejection, empty DataFrame registry handling.

---

## 15. Sample Dataset & Instant Evaluation

CineScope includes a curated, balanced 50-movie sample dataset located at:
```
data/sample/sample_movies.csv
```
This dataset contains complete metadata spanning 1972 through 2023, across genres (Sci-Fi, Drama, Crime, Action, Romance, Animation, Comedy), directors (Christopher Nolan, Quentin Tarantino, Martin Scorsese, Hayao Miyazaki, Bong Joon-ho), and actors.

Users can evaluate the entire application with one click by selecting:
> **"🚀 Load Curated Sample Dataset"** on the Dataset & Data Quality page.

---

## 16. Security, Privacy & Resource Governance

- **Zero External Paid APIs**: The system operates 100% offline using standard open-source Python packages — no OpenAI or TMDB API keys required.
- **Session Isolation**: Uploaded datasets exist solely within the user's private Streamlit session state and are never exposed across concurrent sessions or saved to public directories.
- **Strict Git Boundary (`.gitignore`)**: Raw uploads (`data/raw/*`) and processed exports (`data/processed/*`) are excluded from version control.
- **File Size Ceiling**: A strict 50MB file size guard prevents server resource exhaustion.
- **Sparse Memory Allocation**: Avoids building dense $N \times N$ matrices, ensuring stable deployment on free cloud tiers with strict 1GB RAM ceilings.

---

## 17. Deployment Instructions (Streamlit Community Cloud)

CineScope is configured for zero-friction free deployment to **Streamlit Community Cloud**:

### Deployment Steps:
1. Push this repository to a **Public or Private GitHub Repository**.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.
3. Click **"New App"**.
4. Select your repository: `movie-recommendation-system`.
5. Set the Main file path to: `app.py`.
6. Click **"Deploy!"**.
7. The platform will automatically install packages from `requirements.txt` and launch the application.

---

## 18. Technical Interview & Portfolio Discussion Guide

| Technical Interview Question | CineScope Architectural Response |
| :--- | :--- |
| **Why choose Content-Based over Collaborative Filtering?** | Content-based filtering is ideal for catalog discovery when user interaction histories are unavailable (solving the cold-start problem for users). It is fully explainable and deterministic. |
| **How do you prevent Out-Of-Memory (OOM) crashes on large catalogs?** | We avoid generating dense $N \times N$ similarity matrices. Instead, we preserve the sparse CSR matrix and calculate single-row vector dot products in $O(N)$ space on-the-fly. |
| **How do you handle multi-word entity names (e.g., Tom Hanks vs Tom Hardy)?** | Entity tokenization concatenates names with underscores (`tom_hanks`, `tom_hardy`), preventing spurious unigram overlap on common first names. |
| **How does sublinear TF-IDF scaling benefit movie recommendations?** | Sublinear term frequency ($1 + \log(\text{tf})$) dampens the impact of words repeated multiple times in long narrative synopses, giving equal voice to concise metadata tags. |
| **How do you validate schema compatibility across diverse user datasets?** | A 16-attribute synonym dictionary maps common industry field variants, validated with an automated diagnostic reporter and user-override dropdowns. |
| **What are the mathematical limitations of content-based recommendations?** | Content-based systems suffer from overspecialization (filter bubbles) and cannot recommend serendipitous titles outside the seed movie's metadata proximity without collaborative interaction signals. |

---

## 19. Known Limitations

- **Content-Based Scope**: Does not incorporate user viewing histories or ratings matrices (collaborative filtering).
- **Subjective Enjoyment**: Mathematical similarity reflects descriptive metadata proximity, not a guarantee that a user will personally enjoy the film.
- **Metadata Cold-Start**: Films with empty synopses and missing cast lists rely solely on genre overlap and receive lower-dimensional representations.

---

## 20. Future Improvements

- [ ] **Approximate Nearest Neighbors (ANN)**: Integration with FAISS or HNSW for sub-millisecond retrieval on catalogs exceeding 1,000,000 titles.
- [ ] **Dense Neural Embeddings**: Hybrid feature soup combining TF-IDF with pre-trained Sentence-BERT (`all-MiniLM-L6-v2`) embeddings.
- [ ] **Collaborative Hybridization**: Support for user rating logs via Matrix Factorization (SVD / ALS).

---

## 21. Non-Commercial Scope & Transparency Disclosure

**CineScope** is an analytical data engineering and machine learning portfolio application. It does **not** connect to commercial video-on-demand or streaming services (such as Netflix, Amazon Prime Video, Disney+, or Apple TV+), does **not** hold commercial licensing or streaming availability rights, and does **not** track or collect private personal viewing histories. All analytical conclusions and similarity scores reflect the metadata present in the currently loaded dataset.

---

## 22. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
