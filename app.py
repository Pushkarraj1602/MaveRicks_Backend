import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient, hf_hub_download

# Load environment variables safely
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, "artifacts")

    rf_path = hf_hub_download(repo_id="Satyam-0001/readmission-artifacts", filename="rf_model.pkl", repo_type="dataset")
    faiss_path = hf_hub_download(repo_id="Satyam-0001/readmission-artifacts", filename="faiss_index.bin", repo_type="dataset")

    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)
    with open(os.path.join(artifacts_dir, 'train_outcomes.pkl'), 'rb') as f:
        train_outcomes = pickle.load(f)
    with open(os.path.join(artifacts_dir, 'column_order.pkl'), 'rb') as f:
        column_order = pickle.load(f)
    with open(os.path.join(artifacts_dir, 'default_values.pkl'), 'rb') as f:
        default_values = pickle.load(f)
    faiss_index = faiss.read_index(faiss_path)

    # Initialize HF InferenceClient if token is available
    hf_token = os.getenv("HF_TOKEN")
    hf_client = InferenceClient(token=hf_token) if hf_token else None

    embed_model_path = os.path.join(artifacts_dir, 'sentence_transformer_model')
    embed_model = None
    if os.path.exists(embed_model_path):
        try:
            embed_model = SentenceTransformer(embed_model_path)
            embed_model.max_seq_length = 64
        except Exception:
            embed_model = None

    return rf_model, train_outcomes, column_order, default_values, faiss_index, embed_model, hf_client

rf_model, train_outcomes, column_order, default_values, faiss_index, embed_model, hf_client = load_artifacts()

def get_embedding(text: str) -> np.ndarray:
    if hf_client is not None:
        try:
            emb = hf_client.feature_extraction(text, model=HF_EMBED_MODEL)
            emb = np.array(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = np.expand_dims(emb, axis=0)
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / np.maximum(norm, 1e-12)
        except Exception as e:
            st.warning(f"Hugging Face API failed ({e}), using local model.")
    if embed_model is not None:
        return embed_model.encode([text], normalize_embeddings=True)
    raise RuntimeError("No embedding model available.")

BEST_THRESHOLD = 0.15  # from our threshold sweep on RF+RAG

DIAG_CATEGORIES = ['Circulatory', 'Other', 'Respiratory', 'Digestive', 'Diabetes',
                    'Injury', 'Genitourinary', 'Musculoskeletal', 'Neoplasms']

# ── Helper: build the summary sentence (same logic as training) ──
def make_summary(age, gender, n_inpatient, n_emergency, n_outpatient,
                  diag_1, diag_2, n_medications, time_in_hospital, n_diagnoses):
    return (
        f"{age} year old {gender.lower()} patient, "
        f"{n_inpatient} prior inpatient visits, "
        f"{n_emergency} prior emergency visits, "
        f"{n_outpatient} prior outpatient visits, "
        f"primary diagnosis: {diag_1}, secondary diagnosis: {diag_2}, "
        f"{n_medications} medications, "
        f"{time_in_hospital}-day hospital stay, "
        f"{n_diagnoses} total diagnoses recorded."
    )

# ── Helper: set a one-hot category group correctly (zero others, set chosen) ──
def set_category(row, prefix, value):
    for col in column_order:
        if col.startswith(prefix + "_"):
            row[col] = 0
    target_col = f"{prefix}_{value}"
    if target_col in row:
        row[target_col] = 1
    # if target_col doesn't exist, it means `value` is the dropped reference
    # category from one-hot encoding - leaving everything 0 is correct

# ── Build the UI ──
st.title("Hospital Readmission Risk Predictor")
st.caption("Predicts 30-day readmission risk for diabetic inpatients, using a Random Forest model with a RAG-style retrieved feature.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox("Age", [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
                            format_func=lambda x: f"{x-5} to {x+5}", index=6)
        gender = st.selectbox("Gender", ["Male", "Female"])
        diag_1 = st.selectbox("Primary diagnosis category", DIAG_CATEGORIES)
        time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
        num_lab_procedures = st.slider("Number of lab procedures", 0, 132, 40)
        num_procedures = st.slider("Number of procedures", 0, 6, 1)

    with col2:
        n_inpatient = st.number_input("Prior inpatient visits", 0, 20, 0)
        n_emergency = st.number_input("Prior emergency visits", 0, 20, 0)
        n_outpatient = st.number_input("Prior outpatient visits", 0, 20, 0)
        num_medications = st.slider("Number of medications", 1, 80, 15)
        number_diagnoses = st.slider("Number of diagnoses recorded", 1, 16, 7)
        insulin = st.selectbox("Insulin status", ["No", "Down", "Steady", "Up"])

    submitted = st.form_submit_button("Predict Readmission Risk")

if submitted:
    insulin_map = {'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3}

    # 1. Build summary text (diag_2 defaults to "Other" - not user-collected)
    summary = make_summary(age, gender, n_inpatient, n_emergency, n_outpatient,
                            diag_1, "Other", num_medications, time_in_hospital, number_diagnoses)

    # 2. Embed and retrieve similar patients via FAISS
    query_embedding = get_embedding(summary)
    similarities, neighbor_idx = faiss_index.search(query_embedding, 10)
    similar_rate = train_outcomes[neighbor_idx[0]].mean()

    # 3. Build the full feature row, starting from defaults
    row = dict(default_values)

    row['age'] = age
    row['gender'] = 1 if gender == 'Male' else 0
    row['time_in_hospital'] = time_in_hospital
    row['num_lab_procedures'] = num_lab_procedures
    row['num_procedures'] = num_procedures
    row['num_medications'] = num_medications
    row['number_outpatient'] = n_outpatient
    row['number_emergency'] = n_emergency
    row['number_inpatient'] = n_inpatient
    row['number_diagnoses'] = number_diagnoses
    row['insulin'] = insulin_map[insulin]
    row['similar_case_readmit_rate'] = similar_rate

    set_category(row, 'diag_1_category', diag_1)
    set_category(row, 'diag_2_category', 'Other')

    # 4. Assemble in the exact column order the model expects
    feature_vector = pd.DataFrame([row])[column_order]

    # 5. Predict
    risk_prob = rf_model.predict_proba(feature_vector)[0][1]
    is_high_risk = risk_prob >= BEST_THRESHOLD

    # ── Display results ──
    st.divider()
    st.subheader("Prediction")

    risk_col, gauge_col = st.columns([1, 2])
    with risk_col:
        st.metric("30-day readmission risk", f"{risk_prob*100:.1f}%")
        if is_high_risk:
            st.error("⚠️ High Risk - flagged for review")
        else:
            st.success("✅ Lower Risk")

    st.subheader("Similar past patients (retrieved via RAG)")
    st.write(f"Among the 10 most similar past patients, **{similar_rate*100:.0f}%** were readmitted within 30 days.")

    st.caption(f"Generated patient summary used for retrieval: _{summary}_")

st.divider()
st.caption("This model predicts readmission risk specifically for patients with a diabetes diagnosis during their hospital stay - not a general-purpose readmission predictor.")
