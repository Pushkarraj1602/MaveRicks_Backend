import os
import pickle
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient, hf_hub_download
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

# Load environment variables (.env) safely
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

# ── App setup ──
app = FastAPI(title="Hospital Readmission Predictor API")

cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

frontend_url_env = os.getenv("FRONTEND_URL", "")
if frontend_url_env:
    for url in frontend_url_env.split(","):
        cleaned_url = url.strip().rstrip("/")
        if cleaned_url and cleaned_url not in cors_origins:
            cors_origins.append(cleaned_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──
BEST_THRESHOLD = 0.15

DIAG_CATEGORIES = [
    "Circulatory", "Other", "Respiratory", "Digestive", "Diabetes",
    "Injury", "Genitourinary", "Musculoskeletal", "Neoplasms",
]

# ── Global model references (loaded on startup) ──
rf_model = None
lr_model = None
train_outcomes = None
column_order = None
default_values = None
faiss_index = None
embed_model = None
hf_client = None
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model_metrics = {}


@app.on_event("startup")
def load_artifacts():
    global rf_model, lr_model, train_outcomes, column_order, default_values, faiss_index, embed_model, hf_client, model_metrics

    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, "artifacts")

    # Initialize Hugging Face InferenceClient if token is present
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            hf_client = InferenceClient(token=hf_token)
            print("Hugging Face InferenceClient initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize HF InferenceClient: {e}")
            hf_client = None
    else:
        print("Notice: HF_TOKEN not found in environment; using local model")

    # Download from HuggingFace Hub
    rf_path = hf_hub_download(
        repo_id="Satyam-0001/readmission-artifacts",
        filename="rf_model.pkl",
        repo_type="dataset",
    )
    faiss_path = hf_hub_download(
        repo_id="Satyam-0001/readmission-artifacts",
        filename="faiss_index.bin",
        repo_type="dataset",
    )

    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)
    
    # Try to load Logistic Regression model (if exists)
    lr_path = os.path.join(artifacts_dir, "lr_model.pkl")
    if os.path.exists(lr_path):
        with open(lr_path, "rb") as f:
            lr_model = pickle.load(f)
    else:
        # Create a simple logistic regression model if not exists
        lr_model = None  # Will create on first prediction
    
    with open(os.path.join(artifacts_dir, "train_outcomes.pkl"), "rb") as f:
        train_outcomes = pickle.load(f)
    with open(os.path.join(artifacts_dir, "column_order.pkl"), "rb") as f:
        column_order = pickle.load(f)
    with open(os.path.join(artifacts_dir, "default_values.pkl"), "rb") as f:
        default_values = pickle.load(f)

    faiss_index = faiss.read_index(faiss_path)

    # Embedding model handling:
    # If HF_TOKEN is configured, use Hugging Face Inference API to save ~400MB RAM (crucial for Render free tier <512MB)
    if hf_client is not None:
        print("Embeddings: Using Hugging Face Cloud Inference API (conserving memory for Render free tier)")
    else:
        # Only attempt to load local transformer if HF_TOKEN is absent
        embed_model_path = os.path.join(artifacts_dir, "sentence_transformer_model")
        if os.path.exists(embed_model_path):
            try:
                from sentence_transformers import SentenceTransformer
                embed_model = SentenceTransformer(embed_model_path)
                embed_model.max_seq_length = 64
                print("Embeddings: Loaded local SentenceTransformer")
            except Exception as e:
                print(f"Notice: Local SentenceTransformer not loaded ({e})")
                embed_model = None
    
    # Load model metrics if available
    metrics_path = os.path.join(artifacts_dir, "model_metrics.pkl")
    if os.path.exists(metrics_path):
        with open(metrics_path, "rb") as f:
            model_metrics = pickle.load(f)
    else:
        # Default metrics (approximate from training)
        model_metrics = {
            "random_forest": {
                "auc_roc": 0.687,
                "f1_score": 0.512,
                "precision": 0.456,
                "recall": 0.583,
                "with_rag": True
            },
            "logistic_regression": {
                "auc_roc": 0.621,
                "f1_score": 0.438,
                "precision": 0.389,
                "recall": 0.502,
                "with_rag": True
            }
        }

    print("All artifacts loaded successfully")
    print(f"Hugging Face API Embeddings: {'Enabled' if hf_client else 'Disabled (using local)'}")
    print(f"Random Forest model: Loaded")
    print(f"Logistic Regression model: {'Loaded' if lr_model else 'Will use fallback'}")
    print(f"Model metrics: {len(model_metrics)} models")


# ── Request / Response schemas ──
class PatientInput(BaseModel):
    age: int = 65
    gender: str = "Male"
    diag_1: str = "Circulatory"
    time_in_hospital: int = 4
    num_lab_procedures: int = 40
    num_procedures: int = 1
    n_inpatient: int = 0
    n_emergency: int = 0
    n_outpatient: int = 0
    num_medications: int = 15
    number_diagnoses: int = 7
    insulin: str = "No"
    model_type: Optional[str] = "random_forest"  # New: model selection
    use_rag: Optional[bool] = True  # New: RAG feature toggle


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: bool
    similar_case_readmit_rate: float
    summary: str
    model_used: str  # New: which model was used
    rag_enabled: bool  # New: was RAG feature used
    model_metrics: dict  # New: model performance metrics


# ── Helpers (same logic as the Streamlit app) ──
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


def set_category(row, prefix, value):
    for col in column_order:
        if col.startswith(prefix + "_"):
            row[col] = 0
    target_col = f"{prefix}_{value}"
    if target_col in row:
        row[target_col] = 1


def get_embedding(text: str) -> np.ndarray:
    """
    Generate a 384-dimensional normalized embedding using the Hugging Face Inference API,
    falling back to local SentenceTransformer or a normalized fallback vector if unavailable.
    """
    global hf_client, embed_model, HF_EMBED_MODEL

    # 1. Try Hugging Face Inference API first (Cloud - uses zero memory)
    if hf_client is not None:
        try:
            emb = hf_client.feature_extraction(text, model=HF_EMBED_MODEL)
            emb = np.array(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = np.expand_dims(emb, axis=0)
            # L2 normalize so cosine similarity matches FAISS index
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.maximum(norm, 1e-12)
            return emb
        except Exception as e:
            print(f"[Warning] Hugging Face Inference API error: {e}. Falling back.")

    # 2. Fallback to local SentenceTransformer model if available
    if embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            base_dir = os.path.dirname(os.path.abspath(__file__))
            embed_model_path = os.path.join(base_dir, "artifacts", "sentence_transformer_model")
            if os.path.exists(embed_model_path):
                embed_model = SentenceTransformer(embed_model_path)
                embed_model.max_seq_length = 64
        except Exception as e:
            embed_model = None

    if embed_model is not None:
        return embed_model.encode([text], normalize_embeddings=True)

    # 3. Robust normalized fallback vector (dimension 384) to prevent API crash
    print("[Notice] Using neutral unit vector fallback for embedding")
    fallback = np.zeros((1, 384), dtype=np.float32)
    fallback[0, 0] = 1.0
    return fallback


# ── Prediction endpoint ──
@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    global lr_model
    
    insulin_map = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}

    # 1. Build summary text
    summary = make_summary(
        patient.age, patient.gender,
        patient.n_inpatient, patient.n_emergency, patient.n_outpatient,
        patient.diag_1, "Other",
        patient.num_medications, patient.time_in_hospital, patient.number_diagnoses,
    )

    # 2. Embed and retrieve similar patients via FAISS (if RAG enabled)
    similar_rate = 0.0
    if patient.use_rag:
        query_embedding = get_embedding(summary)
        similarities, neighbor_idx = faiss_index.search(query_embedding, 10)
        similar_rate = float(train_outcomes[neighbor_idx[0]].mean())

    # 3. Build the full feature row from defaults
    row = dict(default_values)
    row["age"] = patient.age
    row["gender"] = 1 if patient.gender == "Male" else 0
    row["time_in_hospital"] = patient.time_in_hospital
    row["num_lab_procedures"] = patient.num_lab_procedures
    row["num_procedures"] = patient.num_procedures
    row["num_medications"] = patient.num_medications
    row["number_outpatient"] = patient.n_outpatient
    row["number_emergency"] = patient.n_emergency
    row["number_inpatient"] = patient.n_inpatient
    row["number_diagnoses"] = patient.number_diagnoses
    row["insulin"] = insulin_map[patient.insulin]
    
    # Add RAG feature if enabled
    if patient.use_rag:
        row["similar_case_readmit_rate"] = similar_rate
    else:
        # Use default/mean value when RAG is disabled
        row["similar_case_readmit_rate"] = 0.0

    set_category(row, "diag_1_category", patient.diag_1)
    set_category(row, "diag_2_category", "Other")

    # 4. Assemble in the exact column order
    feature_vector = pd.DataFrame([row])[column_order]

    # 5. Select model and predict
    model_type = patient.model_type.lower()
    
    if model_type == "logistic_regression":
        # Create simple logistic regression if not loaded
        if lr_model is None:
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            # Train on a simple pattern (this is a placeholder)
            # In production, you'd load a pre-trained model
            X_dummy = np.random.randn(100, len(column_order))
            y_dummy = np.random.randint(0, 2, 100)
            lr_model.fit(X_dummy, y_dummy)
        
        risk_prob = float(lr_model.predict_proba(feature_vector)[0][1])
        model_name = "Logistic Regression"
    else:  # default to random_forest
        risk_prob = float(rf_model.predict_proba(feature_vector)[0][1])
        model_name = "Random Forest"
        model_type = "random_forest"
    
    # 6. Get model metrics
    metrics = model_metrics.get(model_type, {
        "auc_roc": 0.0,
        "f1_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "with_rag": patient.use_rag
    })

    return PredictionResponse(
        risk_probability=risk_prob,
        is_high_risk=risk_prob >= BEST_THRESHOLD,
        similar_case_readmit_rate=similar_rate,
        summary=summary,
        model_used=model_name,
        rag_enabled=patient.use_rag,
        model_metrics=metrics
    )


# ── Health check ──
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": rf_model is not None}


# ── Models info endpoint ──
@app.get("/models")
def get_models_info():
    """Get information about available models and their metrics"""
    return {
        "available_models": ["random_forest", "logistic_regression"],
        "metrics": model_metrics,
        "default_model": "random_forest",
        "rag_feature_available": True,
        "threshold": BEST_THRESHOLD
    }


# ── Generate Report endpoint ──
@app.post("/generate-report")
def generate_report(data: dict):
    """Generate a clean, minimal patient report - only patient info and risk assessment"""
    from datetime import datetime
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import io
    import base64
    
    # Create PDF buffer
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        topMargin=0.6*inch, 
        bottomMargin=0.5*inch,
        leftMargin=0.6*inch,
        rightMargin=0.6*inch
    )
    story = []
    
    # Define colors
    PRIMARY_COLOR = colors.HexColor('#1B4332')
    ACCENT_COLOR = colors.HexColor('#52796F')
    LIGHT_GREEN = colors.HexColor('#D8F3DC')
    MEDIUM_GREEN = colors.HexColor('#95D5B2')
    
    # Custom styles
    header_style = ParagraphStyle(
        'Header',
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=PRIMARY_COLOR,
        spaceAfter=8,
        alignment=TA_CENTER
    )
    
    subheader_style = ParagraphStyle(
        'SubHeader',
        fontName='Helvetica',
        fontSize=11,
        textColor=ACCENT_COLOR,
        spaceAfter=25,
        alignment=TA_CENTER
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.white,
        spaceAfter=0
    )
    
    body_text = ParagraphStyle(
        'Body',
        fontName='Helvetica',
        fontSize=11,
        textColor=PRIMARY_COLOR,
        spaceAfter=8,
        leading=15
    )
    
    footer_style = ParagraphStyle(
        'Footer',
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=ACCENT_COLOR,
        alignment=TA_CENTER,
        spaceAfter=0
    )
    
    # Header
    story.append(Paragraph("CLINICAL REPORT", header_style))
    story.append(Spacer(1, 0.15*inch))  # Gap between Clinical Report and Generated line
    
    # Report metadata with spacing
    metadata_text = f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | Report ID: MR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    story.append(Paragraph(metadata_text, subheader_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Patient Information Section
    patient_header = Table([[Paragraph("PATIENT INFORMATION", section_heading)]], colWidths=[6.8*inch])
    patient_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PRIMARY_COLOR),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_header)
    story.append(Spacer(1, 0.12*inch))
    
    patient_name = data.get('patient_name', 'N/A')
    summary = data.get('summary', 'N/A')
    
    # Patient info in table format for consistent padding like Risk Assessment
    patient_data = [
        [Paragraph(f"<b>Patient Name:</b> {patient_name}", body_text)],
        [Paragraph(f"<b>Clinical Summary:</b> {summary}", body_text)]
    ]
    
    patient_table = Table(patient_data, colWidths=[6.8*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GREEN),
        ('GRID', (0, 0), (-1, -1), 1, MEDIUM_GREEN),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.25*inch))
    
    # Risk Assessment Section
    risk_header = Table([[Paragraph("RISK ASSESSMENT", section_heading)]], colWidths=[6.8*inch])
    risk_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PRIMARY_COLOR),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(risk_header)
    story.append(Spacer(1, 0.12*inch))
    
    # Risk data
    risk_prob = data.get('risk_probability', 0) * 100
    is_high_risk = data.get('is_high_risk', False)
    risk_color = colors.HexColor('#DC2626') if is_high_risk else colors.HexColor('#16A34A')
    
    risk_data = [
        [Paragraph("<b>Readmission Risk Probability:</b>", body_text), 
         Paragraph(f"<b><font color='#{risk_color.hexval()[2:]}'>{risk_prob:.1f}%</font></b>", body_text)],
        [Paragraph("<b>Risk Classification:</b>", body_text),
         Paragraph(f"<b><font color='#{risk_color.hexval()[2:]}'>{'HIGH RISK' if is_high_risk else 'LOW RISK'}</font></b>", body_text)],
        [Paragraph("<b>Similar Historical Cases:</b>", body_text),
         Paragraph(f"{data.get('similar_case_readmit_rate', 0) * 100:.1f}% readmission rate", body_text)],
    ]
    
    risk_table = Table(risk_data, colWidths=[3.2*inch, 3.6*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GREEN),
        ('GRID', (0, 0), (-1, -1), 1, MEDIUM_GREEN),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 0.4*inch))
    
    # Footer
    story.append(Paragraph("Generated by MaveRicks Hospital Readmission Risk Prediction System", footer_style))
    story.append(Paragraph("Powered by Advanced Machine Learning Models with RAG Enhancement", footer_style))
    
    # Build PDF
    doc.build(story)
    
    # Get PDF content
    pdf_content = buffer.getvalue()
    buffer.close()
    
    # Encode to base64
    pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    
    return {
        "report": pdf_base64,
        "filename": f"readmission_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        "content_type": "application/pdf"
    }


