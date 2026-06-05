# 🏛️ Smart Recruitment Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-3.7.4-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)

**An enterprise-grade AI recruitment platform powered by NLP and custom PyTorch Transformer models.**  
Automates candidate evaluation, extracts structured resume intelligence, and delivers explainable hiring decisions.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-recruitment-intelligence-platform.streamlit.app)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Deep Learning Architecture](#-deep-learning-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Local Setup](#-local-setup)
- [Streamlit Cloud Deployment](#-streamlit-cloud-deployment)
- [Module Details](#-module-details)
- [Screenshots](#-screenshots)
- [Author](#-author)

---

## 🎯 Overview

The **Smart Recruitment Intelligence Platform (SRIP)** is a full-stack AI application that automates and enhances the recruitment pipeline using a multi-stage intelligence engine:

- **NLP-powered resume parsing** — extract skills, experience, and education from raw text using spaCy and regex pipelines
- **Semantic similarity ranking** — match candidates against job descriptions with TF-IDF and skill-overlap scoring
- **Custom PyTorch Transformer** — Self-Attention + Positional Encoding for deep candidate-level understanding
- **Explainability module** — attention heatmaps and structured, human-readable hiring reports

---

## 🌐 Live Demo

> Deploy your own instance using the **[Streamlit Cloud Deployment](#-streamlit-cloud-deployment)** guide below.

```
https://<your-slug>.streamlit.app
```

---

## ✨ Features

| # | Module | Description |
|---|--------|-------------|
| 1 | 📊 **Dashboard Overview** | KPI metrics, dataset summary, top candidate records |
| 2 | 📈 **Resume Analytics** | Category/experience distributions, skill word clouds, length analysis |
| 3 | 🔍 **Information Extraction** | NLP-based parsing of skills, education, experience years |
| 4 | 🏆 **Ranking Engine** | Candidate scoring and job-fit ranking via similarity engine |
| 5 | 🧠 **Deep Learning Module** | Self-Attention, Positional Encoding proof, multi-head benchmark |
| 6 | 💡 **Explainability Module** | Attention heatmaps per head + structured hiring reports |

---

## 🧠 Deep Learning Architecture

```
Input Text
    │
    ▼
Tokenisation & Encoding
    │
    ▼
Embedding Layer  (vocab_size × embed_dim)
    │
    ▼
Positional Encoding  (sinusoidal, order-sensitive)
    │
    ▼
Multi-Head Self-Attention  (h = 2, 4, or 8 heads)
    │
    ▼
Layer Normalisation
    │
    ▼
Feed-Forward Layer
    │
    ▼
Linear + Softmax  → Candidate Category Score
```

### Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| **Self-Attention** | Scaled dot-product attention from scratch in PyTorch |
| **Positional Encoding** | Sinusoidal PE proving order-sensitivity in embeddings |
| **Multi-Head Attention** | Benchmarked across 2, 4, 8 heads (100-pass timing) |
| **ResumeClassifier** | End-to-end Transformer for resume category classification |
| **Attention Heatmap** | Per-head visualisation of token attention weights |

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **UI / Frontend** | Streamlit + Custom CSS (Navy/Gold theme) | 1.35.0 |
| **Deep Learning** | PyTorch (custom Transformer) | 2.3.0 |
| **NLP** | spaCy (`en_core_web_sm`) | 3.7.4 |
| **ML Utilities** | scikit-learn | 1.4.2 |
| **Data** | pandas, NumPy | 2.2.2 / 1.26.4 |
| **Visualisation** | matplotlib, seaborn, WordCloud | 3.8.4 / 0.13.2 |
| **PDF Parsing** | PyPDF2 | 3.0.1 |
| **Typography** | EB Garamond + Inter (Google Fonts) | — |

---

## 📁 Project Structure

```
Smart-Recruitment-Intelligence-Platform/
│
├── 📄 app.py                      # Main Streamlit application (all pages)
├── 📄 train.py                    # Model training script (offline)
├── 📄 generate_data.py            # Synthetic resume data generator
├── 📄 requirements.txt            # All Python dependencies
│
├── 📂 .streamlit/
│   └── config.toml                # Professional Navy/Gold Streamlit theme
│
├── 📂 src/
│   ├── analytics.py               # Resume analytics & chart generation
│   ├── extraction.py              # NLP information extraction engine
│   ├── matching.py                # Candidate similarity & ranking engine
│   ├── explainability.py          # Attention heatmaps & hiring reports
│   └── 📂 models/
│       ├── classifier.py          # PyTorch ResumeClassifier (Transformer)
│       ├── attention.py           # MultiHeadAttention (from scratch)
│       └── positional.py         # Positional Encoding + proof demo
│
└── 📂 data/
    └── synthetic_resumes.csv      # Auto-generated on first run (500 records)
```

---

## 🖥️ Local Setup

### Prerequisites

- **Python 3.9+**
- **pip** (or conda)
- **Git**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/BharathReddyRamasani/Smart-Recruitment-Intelligence-Platform.git
cd Smart-Recruitment-Intelligence-Platform

# 2. Create a virtual environment (recommended)
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
# source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

> **💡 Note:** On first run, `generate_data.py` auto-generates `data/synthetic_resumes.csv` (500 synthetic candidates). No manual step needed.

---

## 🚀 Streamlit Cloud Deployment

Deploy this app for **free** in under 5 minutes using [Streamlit Community Cloud](https://streamlit.io/cloud).

### Prerequisites

- A **GitHub account** with this repo pushed (already done ✅)
- A free **Streamlit Community Cloud** account (sign up at [share.streamlit.io](https://share.streamlit.io))

---

### Step-by-Step Deployment Guide

#### Step 1 — Sign in to Streamlit Cloud

Go to 👉 **[share.streamlit.io](https://share.streamlit.io)**  
Click **"Sign in with GitHub"** and authorise Streamlit.

---

#### Step 2 — Create a New App

Click the **"New app"** button (top-right corner).

---

#### Step 3 — Configure Your App

Fill in the deployment form:

| Field | Value |
|-------|-------|
| **Repository** | `BharathReddyRamasani/Smart-Recruitment-Intelligence-Platform` |
| **Branch** | `main` |
| **Main file path** | `app.py` |
| **App URL (slug)** | e.g. `smart-recruitment-ai` *(choose your own)* |

---

#### Step 4 — Deploy

Click **"Deploy!"**

Streamlit Cloud will:
1. Clone your repository
2. Automatically install all packages from `requirements.txt`
3. Download the spaCy model (`en_core_web_sm`) from the wheel URL in requirements
4. Launch your app

---

#### Step 5 — Access Your Live App

Your app is now live at:

```
https://smart-recruitment-ai.streamlit.app
```

*(Replace `smart-recruitment-ai` with the slug you chose)*

---

### ⚠️ Important Deployment Notes

| Item | Status | Notes |
|------|--------|-------|
| `requirements.txt` | ✅ Ready | All deps pinned with exact versions |
| spaCy model | ✅ Included | Installed via wheel URL in requirements.txt |
| Synthetic data | ✅ Auto-generated | Created on first run — no manual step needed |
| Streamlit theme | ✅ Auto-applied | Navy/Gold theme via `.streamlit/config.toml` |
| Model training | ⚠️ Slow on free tier | Free tier has limited CPU — training may take longer |
| `.pth` weights | ⚠️ Ephemeral | Model weights reset on each Streamlit Cloud restart |

---

### Updating the Deployed App

Any push to the `main` branch automatically re-deploys the app:

```bash
git add .
git commit -m "update"
git push origin main
```

---

## 📦 Module Details

### `src/extraction.py` — Information Extraction Engine
Parses raw resume text to extract structured fields:
- **Skills** — keyword matching against a curated domain skill dictionary
- **Experience years** — regex-based numeric detection
- **Education level** — degree keyword classification (BSc / MSc / PhD)

### `src/matching.py` — Similarity & Ranking Engine
Computes a weighted composite score for each candidate vs. a job description:
```
Score = (skill_overlap × 0.60) + (experience_proximity × 0.30)
```

### `src/models/attention.py` — MultiHeadAttention
Pure PyTorch implementation of scaled dot-product multi-head self-attention:
- Supports configurable head counts (2, 4, 8)
- Returns both output and raw attention weight tensors

### `src/models/positional.py` — Positional Encoding
Sinusoidal positional encoding (as in *Attention Is All You Need*):
- Proves order-sensitivity: forward vs. reversed embeddings diverge measurably
- Visualised live in the Deep Learning page

### `src/models/classifier.py` — ResumeClassifier
Full Transformer-based classifier pipeline:
```
Embedding → PositionalEncoding → MultiHeadAttention → LayerNorm → Linear → Softmax
```

### `src/analytics.py` — Resume Analytics
Chart generation for:
- Category distribution (bar/pie)
- Experience distribution (histogram)
- Skill frequency word cloud
- Resume length analysis

### `src/explainability.py` — Explainability Module
- **Attention heatmaps** — per-head token × token weight visualisation using seaborn
- **Hiring report** — structured markdown report with skill match %, experience match %, and recommendation

---

## 👤 Author

**Bharath Reddy Ramasani**  
📧 ramasanibharathreddy2004@gmail.com  
🔗 [GitHub — BharathReddyRamasani](https://github.com/BharathReddyRamasani)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<div align="center">

Built with ❤️ using **Streamlit** · **PyTorch** · **spaCy**

⭐ Star this repo if you found it useful!

</div>