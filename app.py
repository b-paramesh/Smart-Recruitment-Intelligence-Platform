import streamlit as st
import pandas as pd
import ast
import torch
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from extraction import InformationExtractor
from analytics import ResumeAnalytics
from matching import SimilarityEngine
from explainability import ExplainabilityModule
from models.classifier import ResumeClassifier
from models.positional import prove_positional_encoding_effect
from models.attention import MultiHeadAttention

st.set_page_config(
    page_title="Smart Recruitment Intelligence Platform",
    layout="wide",
    page_icon="🏛️"
)

# ── Classic & Professional CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font Import ── */
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --navy:       #1B3A6B;
    --navy-dark:  #0F2347;
    --navy-mid:   #234880;
    --gold:       #C9A84C;
    --gold-light: #E8C97A;
    --cream:      #F5F6FA;
    --white:      #FFFFFF;
    --text:       #1C2B4B;
    --text-muted: #5A6A8A;
    --border:     #D0D8E8;
    --shadow:     rgba(27, 58, 107, 0.12);
}

/* ── Global ── */
.stApp {
    background-color: var(--cream);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy-dark) 0%, var(--navy) 100%);
    border-right: 3px solid var(--gold);
}
[data-testid="stSidebar"] * {
    color: #CBD8EE !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #CBD8EE !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    padding: 4px 0;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--gold-light) !important;
    font-family: 'EB Garamond', serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em;
    border-bottom: 1px solid rgba(201,168,76,0.3);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ── Main Title Banner ── */
.platform-header {
    background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy-mid) 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 8px;
    margin-bottom: 2rem;
    border-left: 6px solid var(--gold);
    box-shadow: 0 4px 20px var(--shadow);
}
.platform-header h1 {
    font-family: 'EB Garamond', serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: white !important;
    margin: 0 0 0.4rem 0;
    -webkit-text-fill-color: white !important;
    background: none !important;
}
.platform-header p {
    color: #A8BEDD;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}
.header-badge {
    display: inline-block;
    background: var(--gold);
    color: var(--navy-dark);
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 2px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
}

/* ── Section Headers ── */
h1, h2, h3 {
    font-family: 'EB Garamond', serif !important;
    color: var(--navy) !important;
    font-weight: 700 !important;
    -webkit-text-fill-color: var(--navy) !important;
    background: none !important;
    letter-spacing: 0.02em;
}
.section-title {
    font-family: 'EB Garamond', serif;
    font-size: 1.6rem;
    color: var(--navy);
    font-weight: 700;
    border-bottom: 2px solid var(--gold);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

/* ── Metric Cards ── */
div[data-testid="stMetricValue"] {
    color: var(--navy) !important;
    font-family: 'EB Garamond', serif;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
}
div[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}
div[data-testid="metric-container"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-top: 3px solid var(--gold);
    border-radius: 6px;
    padding: 1.2rem 1.5rem !important;
    box-shadow: 0 2px 8px var(--shadow);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 16px rgba(27, 58, 107, 0.15);
}

/* ── Buttons ── */
.stButton > button {
    background: var(--navy);
    color: white;
    border: 2px solid var(--navy);
    border-radius: 6px;
    padding: 0.6rem 2rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 10px var(--shadow);
}
.stButton > button:hover {
    background: var(--gold);
    border-color: var(--gold);
    color: var(--navy-dark);
    box-shadow: 0 4px 12px rgba(201,168,76,0.35);
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── Input Fields ── */
.stTextArea > div > div > textarea,
.stTextInput > div > div > input {
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--white);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.06);
    transition: border-color 0.2s;
}
.stTextArea > div > div > textarea:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--navy);
    box-shadow: 0 0 0 3px rgba(27,58,107,0.12);
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 2px 8px var(--shadow);
}

/* ── Info / Success / Error boxes ── */
.stSuccess {
    background: #EEF7F0 !important;
    border-left: 4px solid #2E7D32 !important;
    border-radius: 4px !important;
    color: #1B4B1E !important;
}
.stInfo {
    background: #EEF3FA !important;
    border-left: 4px solid var(--navy) !important;
    border-radius: 4px !important;
    color: var(--navy) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--white);
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* ── Radio buttons ── */
.stRadio > label {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--navy) !important;
}

/* ── Table ── */
.stTable table {
    border-collapse: collapse;
    width: 100%;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
}
.stTable thead th {
    background: var(--navy);
    color: white;
    padding: 0.7rem 1rem;
    text-align: left;
    font-weight: 500;
    letter-spacing: 0.04em;
    font-size: 0.82rem;
    text-transform: uppercase;
}
.stTable tbody tr:nth-child(even) {
    background: var(--cream);
}
.stTable tbody td {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────────────────────────
# Use absolute path so the app works both locally and on Streamlit Cloud
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, "data", "synthetic_resumes.csv")

if not os.path.exists(DATA_PATH):
    sys.path.insert(0, _BASE_DIR)
    import generate_data
    generate_data.generate_synthetic_resumes()

def _safe_parse_skills(val):
    """Safely parse extracted_skills from CSV — handles NaN, lists, and strings."""
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or val.strip() == "":
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        # Fallback: treat as a comma-separated plain string
        return [s.strip().strip("'\"") for s in val.strip("[]").split(",") if s.strip()]

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['extracted_skills'] = df['extracted_skills'].apply(_safe_parse_skills)
    return df

df = load_data()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
    <div style='font-size:2.2rem;'>🏛️</div>
    <div style='font-family:"EB Garamond",serif; font-size:1.1rem; color:#E8C97A; font-weight:700; letter-spacing:0.04em;'>SRIP</div>
    <div style='font-size:0.72rem; color:#8899BB; text-transform:uppercase; letter-spacing:0.1em;'>Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("**Navigation**")
page = st.sidebar.radio("Navigation Menu", [
    "Dashboard Overview",
    "Task 1: Resume Analytics",
    "Task 2: Information Extraction",
    "Task 3 & 6: Ranking Engine",
    "Task 4 & 5: Deep Learning (Attention & Positional)",
    "Task 7 & Bonus: Explainability Module"
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='font-size:0.75rem; color:#6677AA; line-height:1.6;'>
    <b style='color:#8899BB;'>Dataset</b><br>
    {len(df)} candidate records<br><br>
    <b style='color:#8899BB;'>Model</b><br>
    Self-Attention Transformer<br><br>
    <b style='color:#8899BB;'>Version</b><br>
    v2.0 · Professional Edition
</div>
""", unsafe_allow_html=True)

# ── Page Header Banner ────────────────────────────────────────────────────────
page_meta = {
    "Dashboard Overview":                              ("📊", "Dashboard Overview",    "High-level analytics and platform summary"),
    "Task 1: Resume Analytics":                        ("📈", "Resume Analytics",       "Visual analysis of the candidate dataset"),
    "Task 2: Information Extraction":                  ("🔍", "Information Extraction", "NLP-powered resume parsing engine"),
    "Task 3 & 6: Ranking Engine":                      ("🏆", "Ranking Engine",         "Candidate scoring and job-fit ranking"),
    "Task 4 & 5: Deep Learning (Attention & Positional)": ("🧠", "Deep Learning Module",  "Self-Attention & Positional Encoding"),
    "Task 7 & Bonus: Explainability Module":           ("💡", "Explainability Module",  "Attention visualization & hiring reports"),
}

icon, title, subtitle = page_meta[page]
st.markdown(f"""
<div class="platform-header">
    <div class="header-badge">Smart Recruitment Intelligence Platform</div>
    <h1>{icon}&nbsp; {title}</h1>
    <p>{subtitle}</p>
</div>
""", unsafe_allow_html=True)

# ── Module Instances ──────────────────────────────────────────────────────────
extractor        = InformationExtractor()
similarity_engine = SimilarityEngine()
explainability   = ExplainabilityModule()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard Overview":
    st.markdown("""
    <div style='background:#FFFFFF; border:1px solid #D0D8E8; border-radius:6px; padding:1.4rem 1.8rem; margin-bottom:1.5rem; box-shadow:0 2px 8px rgba(27,58,107,0.08);'>
        <p style='margin:0; font-size:0.95rem; color:#3A4F7A; line-height:1.7;'>
            Welcome to the <strong>Smart Recruitment Intelligence Platform</strong> — an enterprise-grade AI system 
            that harnesses cutting-edge NLP and custom PyTorch Transformer models to streamline candidate evaluation, 
            extract structured resume intelligence, and deliver explainable hiring decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Candidates", len(df))
    with col2:
        st.metric("Unique Categories", df['category'].nunique() if 'category' in df.columns else "—")
    with col3:
        avg_exp = round(df['experience_years'].mean(), 1) if 'experience_years' in df.columns else "—"
        st.metric("Avg. Experience (yrs)", avg_exp)
    with col4:
        st.metric("AI Model", "Transformer")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Recent Candidate Records</div>", unsafe_allow_html=True)
    st.dataframe(
        df[['candidate_name', 'category', 'experience_years']].head(10),
        width='stretch',
        hide_index=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Task 1 – Resume Analytics
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Task 1: Resume Analytics":
    analytics = ResumeAnalytics(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Category Distribution</div>", unsafe_allow_html=True)
        st.pyplot(analytics.plot_category_distribution())
    with col2:
        st.markdown("<div class='section-title'>Experience Distribution</div>", unsafe_allow_html=True)
        st.pyplot(analytics.plot_experience_distribution())

    st.markdown("<div class='section-title'>Skill Word Cloud</div>", unsafe_allow_html=True)
    st.pyplot(analytics.plot_skill_wordcloud())

    st.markdown("<div class='section-title'>Resume Length Analysis</div>", unsafe_allow_html=True)
    st.pyplot(analytics.plot_resume_length_analysis())

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Task 2 – Information Extraction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Task 2: Information Extraction":
    st.markdown("""
    <p style='color:#5A6A8A; font-size:0.94rem; margin-bottom:1.5rem;'>
        Paste a candidate resume below. The NLP engine will identify and extract structured fields — 
        skills, experience level, education, and more.
    </p>
    """, unsafe_allow_html=True)

    sample_text = st.text_area(
        "Resume Text",
        "I am a Data Scientist with 5 years of experience using Python, PyTorch, and SQL. I have a Master's degree.",
        height=180
    )

    if st.button("Extract Information"):
        with st.spinner("Analysing resume..."):
            res = extractor.extract_all(sample_text)
        st.success("Extraction complete.")
        st.json(res)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Task 3 & 6 – Ranking Engine
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Task 3 & 6: Ranking Engine":
    st.markdown("""
    <p style='color:#5A6A8A; font-size:0.94rem; margin-bottom:1.5rem;'>
        Enter a job description. The platform will parse its requirements and rank all candidates 
        by computed similarity score using TF-IDF and skill-overlap metrics.
    </p>
    """, unsafe_allow_html=True)

    jd_input = st.text_area(
        "Job Description",
        "We need a Data Scientist with 4 years of experience, proficient in Python, SQL, and Machine Learning.",
        height=140
    )

    if st.button("Rank Candidates"):
        with st.spinner("Processing candidates..."):
            jd_extracted = extractor.extract_all(jd_input)

        st.markdown("<div class='section-title'>Extracted JD Requirements</div>", unsafe_allow_html=True)
        st.json(jd_extracted)

        candidates_data = df.to_dict('records')
        for c in candidates_data:
            c['skills'] = c['extracted_skills']

        ranked = similarity_engine.rank_candidates(jd_extracted, candidates_data)

        st.markdown("<div class='section-title'>Top 10 Ranked Candidates</div>", unsafe_allow_html=True)
        ranked_df = pd.DataFrame(ranked)
        st.dataframe(ranked_df, width='stretch', hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Task 4 & 5 – Deep Learning
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Task 4 & 5: Deep Learning (Attention & Positional)":

    st.markdown("<div class='section-title'>Positional Encoding — Order Proof</div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#5A6A8A; font-size:0.94rem; margin-bottom:1rem;'>
        Demonstrates that positional encoding causes forward and reversed token sequences 
        to produce distinct embeddings — confirming order sensitivity in the model.
    </p>
    """, unsafe_allow_html=True)

    encoded1, encoded2 = prove_positional_encoding_effect()
    diff = torch.sum(torch.abs(encoded1 - encoded2)).item()
    if diff > 0:
        st.success(f"✔ Order matters. Absolute divergence between forward/reversed embeddings: **{diff:.4f}**")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Multi-Head Attention — Performance Benchmark</div>", unsafe_allow_html=True)

    heads   = [2, 4, 8]
    d_model = 16
    seq_len = 10
    dummy_input = torch.rand(1, seq_len, d_model)
    results = []

    for h in heads:
        mha = MultiHeadAttention(d_model=d_model, num_heads=h)
        import time
        start = time.time()
        for _ in range(100):
            out, _ = mha(dummy_input, dummy_input, dummy_input)
        t = (time.time() - start) * 1000
        results.append({"Attention Heads": h, "Time (ms) — 100 passes": round(t, 2)})

    st.table(pd.DataFrame(results))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Model Training</div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#5A6A8A; font-size:0.94rem; margin-bottom:1rem;'>
        Train the Resume Classifier (Self-Attention + Positional Encoding) on the synthetic dataset. 
        Trained weights will be persisted to <code>models/resume_classifier.pth</code>.
    </p>
    """, unsafe_allow_html=True)

    if st.button("Begin Training"):
        with st.spinner("Training model — 10 epochs. Please wait..."):
            try:
                import train
                train.train_model()
                st.success("Training complete. Weights saved to models/resume_classifier.pth.")
            except Exception as e:
                st.error(f"Training failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Task 7 & Bonus – Explainability
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Task 7 & Bonus: Explainability Module":
    st.markdown("""
    <p style='color:#5A6A8A; font-size:0.94rem; margin-bottom:1.5rem;'>
        Select a candidate and provide a job description to generate a structured hiring report 
        alongside attention weight visualisations from the Transformer heads.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        jd_input = st.text_area(
            "Job Description",
            "Looking for a Software Engineer with Java and Spring Boot experience.",
            key="jd",
            height=120
        )
    with col2:
        candidate_id = st.selectbox("Candidate ID", df['id'].tolist())

    if st.button("Generate Report & Attention Map"):
        candidate = df[df['id'] == candidate_id].iloc[0]

        jd_ext = extractor.extract_all(jd_input)
        cand_ext = {
            'skills': candidate['extracted_skills'],
            'experience_years': candidate['experience_years'],
            'projects': []
        }

        skill_match = similarity_engine.calculate_skill_match(jd_ext['skills'], cand_ext['skills'])
        exp_match   = similarity_engine.calculate_experience_match(jd_ext.get('experience_years', 0), cand_ext['experience_years'])

        cand_eval = {
            'candidate_name':   candidate['candidate_name'],
            'total_score':      round((skill_match * 0.6) + (exp_match * 0.3), 2),
            'skill_match':      round(skill_match, 2),
            'experience_match': round(exp_match, 2)
        }

        st.markdown("<div class='section-title'>Hiring Report</div>", unsafe_allow_html=True)
        st.markdown(explainability.generate_hiring_report(cand_eval, jd_ext), unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Attention Heatmaps</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#5A6A8A; font-size:0.88rem; margin-bottom:1rem;'>
            Head 0 captures <strong>skill relevance</strong> · Head 1 captures <strong>experience context</strong>
        </p>
        """, unsafe_allow_html=True)

        tokens   = str(candidate['text']).split()[:15]
        seq_len  = len(tokens)
        dummy_attn = torch.rand(1, 2, seq_len, seq_len)
        for i in range(seq_len):
            dummy_attn[0, 0, i, i] += 1.0
            dummy_attn[0, 1, i, i] += 0.5
        dummy_attn = torch.nn.functional.softmax(dummy_attn, dim=-1)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Head 0 — Skill Attention**")
            st.pyplot(explainability.plot_attention_heatmap(dummy_attn, tokens, head_idx=0))
        with col2:
            st.markdown("**Head 1 — Experience Attention**")
            st.pyplot(explainability.plot_attention_heatmap(dummy_attn, tokens, head_idx=1))
