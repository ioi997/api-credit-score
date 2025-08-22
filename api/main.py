from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import shap
from pathlib import Path

# 📌 RGPD : Cet outil respecte le RGPD :
# - Les données sont anonymisées et temporaires
# - Aucune information personnelle n'est stockée
# - Le score est généré à des fins d’évaluation du crédit uniquement

app = FastAPI(title="API de Scoring Crédit", version="0.1.0")

# 🔹 Résolution robuste des chemins vers les modèles
BASE_DIR = Path(__file__).resolve().parent.parent  # ← remonte au dossier racine
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"

# 🔹 Chargement du modèle et du scaler
if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise RuntimeError("❌ Le modèle ou le scaler est manquant. Veuillez les entraîner et sauvegarder.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# 🔹 Noms des features utilisées (dans l’ordre)
FEATURE_NAMES = [
    'DAYS_BIRTH',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'DAYS_EMPLOYED',
    'BUREAU_CREDIT_MEAN',
    'BUREAU_ACTIVE_COUNT',
    'PREV_APP_MEAN',
    'PREV_CREDIT_MEAN',
    'LATE_PAYMENT_RATE'
]

EXPECTED_NB_FEATURES = len(FEATURE_NAMES)

# 🔹 Initialisation de SHAP explainer
explainer = shap.Explainer(model)

# 🔹 Modèle de requête attendu
class ClientData(BaseModel):
    features: List[float]

# 🔹 Route d'accueil
@app.get("/")
def home():
    return {
        "message": "Bienvenue sur l'API de Scoring Crédit 🚀",
        "nb_features_attendus": EXPECTED_NB_FEATURES,
        "endpoint_prediction": "/predict",
        "endpoint_noms_features": "/features-names"
    }

# 🔹 Route pour exposer les noms des variables
@app.get("/features-names")
def get_feature_names():
    return {"features_names": FEATURE_NAMES}

# 🔹 Route de prédiction
@app.post("/predict")
def predict_score(data: ClientData):
    # ✅ Vérification du nombre de variables
    if len(data.features) != EXPECTED_NB_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"❌ Mauvais nombre de features. Attendu : {EXPECTED_NB_FEATURES}, reçu : {len(data.features)}"
        )

    try:
        # 🔹 Préparation des données
        X_input = np.array(data.features).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        # 🔹 Prédiction
        score = model.predict_proba(X_scaled)[0][1]

        # 🔹 Interprétation SHAP
        shap_values = explainer(X_scaled)
        shap_impact = dict(zip(FEATURE_NAMES, shap_values.values[0].tolist()))

        return {
            "score": round(float(score), 4),
            "explication_shap": shap_impact
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prédiction : {str(e)}"
        )
