# 🏛️ Smart Recruitment Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**An enterprise-grade AI recruitment platform powered by NLP and custom PyTorch Transformer models.**  
Streamlines candidate evaluation, extracts structured resume intelligence, and delivers explainable hiring decisions.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-recruitment-intelligence-platform.streamlit.app)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [Module Details](#module-details)

---

## Overview

The **Smart Recruitment Intelligence Platform (SRIP)** is a full-stack AI application that automates and enhances the recruitment pipeline using:

- **NLP-powered resume parsing** to extract skills, experience, and education
- **Semantic similarity ranking** to match candidates against job descriptions
- **Custom PyTorch Transformer** (Self-Attention + Positional Encoding) for deep candidate understanding
- **Explainability module** with attention heatmaps and hiring reports

---

## ✨ Features

| Module | Description |
|--------|-------------|
| 📊 **Dashboard** | High-level KPIs and candidate dataset overview |
| 📈 **Resume Analytics** | Category/experience distributions, skill word clouds |
| 🔍 **Information Extraction** | NLP-based parsing of skills, education, experience |
| 🏆 **Ranking Engine** | TF-IDF + skill-overlap scoring for job-fit ranking |
| 🧠 **Deep Learning** | Self-Attention & Positional Encoding with benchmarks |
| 💡 **Explainability** | Attention heatmaps + structured hiring reports |

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io) with custom CSS (Navy/Gold professional theme)
- **NLP:** scikit-learn, spaCy / regex-based extraction
- **Deep Learning:** PyTorch (custom MultiHeadAttention, PositionalEncoding, ResumeClassifier)
- **Data:** pandas, matplotlib, WordCloud
- **Fonts:** EB Garamond + Inter (Google Fonts)

---

## 📁 Project Structure

```
Smart-Recruitment-Intelligence-Platform/
│
├── app.py                    # Main Streamlit application
├── train.py                  # Model training script
├── generate_data.py          # Synthetic resume data generator
├── requirements.txt          # Python dependencies
│
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
│
├── src/
│   ├── analytics.py          # Resume analytics & visualizations
│   ├── extraction.py         # NLP information extraction engine
│   ├── matching.py           # Candidate similarity & ranking
│   ├── explainability.py     # Attention visualization & reports
│   └── models/
│       ├── classifier.py     # PyTorch ResumeClassifier (Transformer)
│       ├── attention.py      # MultiHeadAttention implementation
│       └── positional.py     # Positional Encoding module
│
└── data/
    └── synthetic_resumes.csv # Auto-generated on first run
```

---

## 🖥️ Local Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/BharathReddyRamasani/Smart-Recruitment-Intelligence-Platform.git
cd Smart-Recruitment-Intelligence-Platform

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The app will automatically open at **http://localhost:8501**

> **Note:** On first run, `generate_data.py` will auto-generate `data/synthetic_resumes.csv` if it doesn't exist.

---

## 🚀 Streamlit Cloud Deployment

Deploy this app for free in minutes using [Streamlit Community Cloud](https://streamlit.io/cloud).

### Step-by-Step Guide

1. **Fork or push this repo to your GitHub account**

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

3. **Click "New app"** and fill in:

   | Field | Value |
   |-------|-------|
   | Repository | `BharathReddyRamasani/Smart-Recruitment-Intelligence-Platform` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | App URL | *(choose your custom slug)* |

4. **Click "Deploy!"** — Streamlit Cloud will install `requirements.txt` automatically

5. Your app will be live at:  
   `https://<your-slug>.streamlit.app`

### Important Notes for Deployment

- ✅ `requirements.txt` includes all dependencies — no extra setup needed
- ✅ `data/synthetic_resumes.csv` is auto-generated on first run
- ✅ `.streamlit/config.toml` sets the professional navy/gold theme automatically
- ⚠️ **Model training** (`Train Model Now` button) requires compute — expect slower performance on free tier
- ⚠️ Trained model weights (`.pth` files) are not persisted between Streamlit Cloud restarts (ephemeral filesystem)

---

## 📦 Module Details

### `src/extraction.py` — Information Extraction Engine
Parses raw resume text to extract:
- **Skills** (keyword matching against a curated skill dictionary)
- **Experience years** (regex-based detection)
- **Education level** (degree keyword matching)

### `src/matching.py` — Similarity & Ranking Engine
Scores candidates against a job description using:
- Skill overlap ratio
- Experience proximity scoring
- Weighted composite score (60% skill, 30% experience)

### `src/models/attention.py` — MultiHeadAttention
Custom PyTorch implementation of scaled dot-product multi-head self-attention.

### `src/models/positional.py` — Positional Encoding
Sinusoidal positional encoding demonstrating order-sensitivity in Transformer architectures.

### `src/models/classifier.py` — ResumeClassifier
End-to-end Transformer classifier:  
`Embedding → PositionalEncoding → MultiHeadAttention → LayerNorm → Linear → Softmax`

---

## 👤 Author

**Bharath Reddy Ramasani**  
📧 ramasanibharathreddy2004@gmail.com  
🔗 [GitHub](https://github.com/BharathReddyRamasani)

---

<div align="center">
<sub>Built with ❤️ using Streamlit & PyTorch</sub>
</div>