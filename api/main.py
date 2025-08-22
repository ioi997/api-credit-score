from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import os
import numpy as np
import joblib

STREAMLIT_URL = os.getenv("STREAMLIT_URL", "https://streamlit-dashboard-ejrg.onrender.com")
USE_SHAP = os.getenv("USE_SHAP", "0") == "1"  # par défaut OFF (performance)
MODEL_DIR = os.getenv("MODEL_DIR")

app = FastAPI(title="API de Scoring Crédit", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[STREAMLIT_URL],  # sécurise le CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(MODEL_DIR) if MODEL_DIR else Path(__file__).resolve().parents[1]
MODEL_PATH = (BASE_DIR / "model" / "model.pkl").resolve()
SCALER_PATH = (BASE_DIR / "model" / "scaler.pkl").resolve()
BACKGROUND_PATH = (BASE_DIR / "model" / "background.npy").resolve()  

FEATURE_NAMES = [
    "DAYS_BIRTH", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED",
    "BUREAU_CREDIT_MEAN", "BUREAU_ACTIVE_COUNT", "PREV_APP_MEAN", "PREV_CREDIT_MEAN", "LATE_PAYMENT_RATE",
]
EXPECTED_NB_FEATURES = len(FEATURE_NAMES)

DEFAULTS_SIMPLE = {
    "BUREAU_CREDIT_MEAN": 0.0,
    "BUREAU_ACTIVE_COUNT": 0.0,
    "PREV_APP_MEAN": 0.0,
    "PREV_CREDIT_MEAN": 0.0,
    "LATE_PAYMENT_RATE": 0.0,
}

_model = None
_scaler = None
_explainer = None
_shap_ready = False

def load_model_once():
    global _model, _scaler
    if _model is not None and _scaler is not None:
        return
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise RuntimeError(f"❌ Modèle/scaler introuvables.\nMODEL_PATH={MODEL_PATH}\nSCALER_PATH={SCALER_PATH}")
    _model = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)

def try_load_shap_once():
    global _explainer, _shap_ready
    if _shap_ready or not USE_SHAP:
        return
    try:
        import shap
        if BACKGROUND_PATH.exists():
            background = np.load(BACKGROUND_PATH)
            if background.ndim == 2 and background.shape[0] > 200:
                background = background[:200]
        else:
            background = shap.maskers.Independent(np.zeros((1, EXPECTED_NB_FEATURES)))
        _explainer = shap.Explainer(model=_model, masker=background, feature_names=FEATURE_NAMES)
        _shap_ready = True
    except Exception:
        _explainer = None
        _shap_ready = False

def predict_proba_safe(X_scaled: np.ndarray) -> float:
    if hasattr(_model, "predict_proba"):
        return float(_model.predict_proba(X_scaled)[0][1])
    if hasattr(_model, "decision_function"):
        raw = float(_model.decision_function(X_scaled)[0])
        return 1.0 / (1.0 + np.exp(-raw))
    pred = int(_model.predict(X_scaled)[0])
    return 0.9 if pred == 1 else 0.1

def compute_shap_impact(X_scaled: np.ndarray) -> Optional[dict]:
    if not _shap_ready or _explainer is None:
        return None
    try:
        shap_values = _explainer(X_scaled)
        values = np.array(getattr(shap_values, "values", shap_values))[0]
        return {name: float(val) for name, val in zip(FEATURE_NAMES, values)}
    except Exception:
        return None

class ClientData(BaseModel):
    features: List[float] = Field(..., description=f"Liste de {EXPECTED_NB_FEATURES} features (ordre fixe).")

class SimpleInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., ge=0)
    credit_amount: float = Field(..., ge=0)
    annuity: float = Field(..., ge=0)
    seniority_years: float = Field(0, ge=0)

def simple_to_features(s: SimpleInput) -> List[float]:
    days_birth = -int(s.age) * 365
    days_employed = -float(s.seniority_years) * 365.0
    feat_map = {
        "DAYS_BIRTH": days_birth,
        "AMT_INCOME_TOTAL": float(s.annual_income),
        "AMT_CREDIT": float(s.credit_amount),
        "AMT_ANNUITY": float(s.annuity),
        "DAYS_EMPLOYED": days_employed,
        "BUREAU_CREDIT_MEAN": DEFAULTS_SIMPLE["BUREAU_CREDIT_MEAN"],
        "BUREAU_ACTIVE_COUNT": DEFAULTS_SIMPLE["BUREAU_ACTIVE_COUNT"],
        "PREV_APP_MEAN": DEFAULTS_SIMPLE["PREV_APP_MEAN"],
        "PREV_CREDIT_MEAN": DEFAULTS_SIMPLE["PREV_CREDIT_MEAN"],
        "LATE_PAYMENT_RATE": DEFAULTS_SIMPLE["LATE_PAYMENT_RATE"],
    }
    return [feat_map[n] for n in FEATURE_NAMES]

@app.get("/")
def root():
    return {
        "message": "Bienvenue sur l'API de Scoring Crédit 🚀",
        "version": app.version,
        "use_shap": USE_SHAP,
        "endpoints": ["/healthz", "/features-names", "/predict", "/predict-simple"],
    }

@app.get("/healthz")
def healthz():
    try:
        load_model_once()
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features-names")
def get_feature_names():
    return {"features_names": FEATURE_NAMES}

@app.post("/predict")
def predict(data: ClientData):
    if len(data.features) != EXPECTED_NB_FEATURES:
        raise HTTPException(status_code=400, detail=f"Mauvais nombre de features: {len(data.features)} (attendu {EXPECTED_NB_FEATURES})")
    try:
        load_model_once()
        X_input = np.array(data.features, dtype=float).reshape(1, -1)
        X_scaled = _scaler.transform(X_input)
        score = predict_proba_safe(X_scaled)
        if USE_SHAP:
            try_load_shap_once()
        shap_impact = compute_shap_impact(X_scaled)
        return {"score": round(float(score), 4), "explication_shap": shap_impact}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

@app.post("/predict-simple")
def predict_simple(s: SimpleInput):
    try:
        load_model_once()
        feats = simple_to_features(s)
        X_input = np.array(feats, dtype=float).reshape(1, -1)
        X_scaled = _scaler.transform(X_input)
        score = predict_proba_safe(X_scaled)
        if USE_SHAP:
            try_load_shap_once()
        shap_impact = compute_shap_impact(X_scaled)
        return {
            "input_features": dict(zip(FEATURE_NAMES, feats)),
            "score": round(float(score), 4),
            "explication_shap": shap_impact,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")
