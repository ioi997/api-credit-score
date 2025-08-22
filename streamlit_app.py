# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go

# ========== CONFIG ==========
st.set_page_config(page_title="💳 Dashboard Néo-Banque", layout="wide")
API_URL = os.environ.get("API_URL", "http://localhost:8000")
BASE_DIR = Path(__file__).resolve().parent
META_PATH = BASE_DIR / "model" / "artifacts" / "model_meta.json"

# ========== STYLES ==========
st.markdown("""
<style>
    .title-center { text-align:center; margin-bottom: 0px; }
    .subtitle-center { text-align:center; color:#6c757d; font-size:14px; margin-top:-8px; }
    .main-button > button {
        background-color: #1a73e8 !important;
        color: white !important;
        font-weight: bold;
        padding: 0.6em 1.2em !important;
        border-radius: 8px !important;
        margin-top: 12px;
    }
    .reset-button > button {
        background-color: #e0e0e0 !important;
        color: black !important;
        font-weight: normal;
        padding: 0.3em 1em !important;
        border-radius: 6px !important;
        margin-left: 0.5em;
    }
    .expander-box {
        background-color: #000000;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========== UTILS ==========
def read_meta():
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def get_readable_name(name):
    return {
        "DAYS_BIRTH": "Âge",
        "AMT_INCOME_TOTAL": "Revenu",
        "AMT_CREDIT": "Crédit",
        "AMT_ANNUITY": "Annuité",
        "DAYS_EMPLOYED": "Ancienneté",
        "BUREAU_CREDIT_MEAN": "Score bureau",
        "BUREAU_ACTIVE_COUNT": "Crédits actifs",
        "PREV_APP_MEAN": "Demandes précédentes",
        "PREV_CREDIT_MEAN": "Crédits précédents",
        "LATE_PAYMENT_RATE": "Retard de paiement",
    }.get(name, name)

# ========== JAUGE ==========
def gauge(prob_default, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_default * 100.0,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, threshold * 100.0], "color": "lightgreen"},
                {"range": [threshold * 100.0, 100], "color": "tomato"},
            ],
        },
    ))
    fig.update_layout(height=250, margin=dict(t=10, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

# ========== UI PRINCIPALE ==========
st.markdown("<h3 class='title-center'>💳 Dashboard Néo-Banque</h3>", unsafe_allow_html=True)

# ✅ DÉCLARATION DES COLONNES AVANT UTILISATION
col_form, col_result = st.columns([1.3, 1.0])

# ========== FORMULAIRE ==========
with col_form:
    st.markdown("### 🧍 Informations Client")

    # Champs obligatoires
    age = st.slider("Âge", 18, 100, 35)
    income = st.number_input("💰 Revenu annuel (€)", min_value=1000, value=25000)
    credit = st.number_input("🏦 Montant du crédit (€)", min_value=500, value=15000)
    annuity = st.number_input("📅 Montant des annuités (€)", min_value=100, value=800)
    years_emp = st.slider("🧑‍💼 Ancienneté (années)", 0, 40, 5)

    facultatifs = {}

    # Champs facultatifs dans une section noire
    with st.expander("🔧 Options supplémentaires (facultatif)", expanded=False):
        st.markdown("<div class='expander-box'>", unsafe_allow_html=True)
        bureau_credit_mean = st.number_input("📊 Moyenne score bureau (€)", value=0)
        if bureau_credit_mean > 0:
            facultatifs["BUREAU_CREDIT_MEAN"] = bureau_credit_mean

        bureau_active_count = st.number_input("📁 Nombre de crédits actifs", value=0)
        if bureau_active_count > 0:
            facultatifs["BUREAU_ACTIVE_COUNT"] = bureau_active_count

        prev_app_mean = st.number_input("📄 Moyenne des précédentes demandes", value=0)
        if prev_app_mean > 0:
            facultatifs["PREV_APP_MEAN"] = prev_app_mean

        prev_credit_mean = st.number_input("💼 Moyenne des crédits précédents", value=0)
        if prev_credit_mean > 0:
            facultatifs["PREV_CREDIT_MEAN"] = prev_credit_mean

        late_payment_rate = st.slider("⏱️ Taux de retard de paiement", 0.0, 1.0, 0.0)
        if late_payment_rate > 0:
            facultatifs["LATE_PAYMENT_RATE"] = late_payment_rate
        st.markdown("</div>", unsafe_allow_html=True)

    # CTA principal + reset
    col_btn, col_reset = st.columns([1, 1])
    with col_btn:
        submitted = st.button("💡 Lancer le scoring", type="primary", use_container_width=True, key="score_btn")
    with col_reset:
        reset = st.button("🔁 Réinitialiser le formulaire", use_container_width=True, key="reset_btn")

# ========== RÉSULTATS ==========
with col_result:
    st.markdown("### 📊 Résultat")

    if submitted:
        payload = {
            "features": [
                -age * 365,
                income,
                credit,
                annuity,
                -years_emp * 365
            ]
        }

        facultative_values = [
            facultatifs.get("BUREAU_CREDIT_MEAN", 0),
            facultatifs.get("BUREAU_ACTIVE_COUNT", 0),
            facultatifs.get("PREV_APP_MEAN", 0),
            facultatifs.get("PREV_CREDIT_MEAN", 0),
            facultatifs.get("LATE_PAYMENT_RATE", 0),
        ]
        payload["features"].extend(facultative_values)

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            score = float(result.get("score", 0.0))
            shap_impact = result.get("explication_shap", {})
            threshold = float(read_meta().get("decision_threshold", 0.5))

            st.markdown("#### 🎯 Score de Risque")
            gauge(score, threshold)

            if score >= threshold:
                st.error(f"❌ Client à risque élevé — score = {score*100:.2f}%")
            else:
                st.success(f"✅ Client solvable — score = {score*100:.2f}%")

            st.markdown("#### 🔍 Interprétation SHAP")
            sorted_items = sorted(shap_impact.items(), key=lambda x: abs(x[1][0]), reverse=True)
            labels = [get_readable_name(k) for k, _ in sorted_items]
            values = [round(v[0], 4) for _, v in sorted_items]

            fig = go.Figure(go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker=dict(color=["green" if v > 0 else "red" for v in values])
            ))
            fig.update_layout(height=400, title="Variables contributives (SHAP)")
            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Erreur lors de l'appel à l’API : {e}")
        except Exception as e:
            st.error(f"⚠️ Erreur inattendue : {e}")
    else:
        st.info("Complétez les champs à gauche puis cliquez sur 💡 Lancer le scoring.")
