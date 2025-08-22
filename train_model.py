import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# 📌 RGPD : Cet outil respecte le RGPD :
# - Les données sont anonymisées et temporaires
# - Aucune information personnelle n'est stockée
# - Le score est généré à des fins d’évaluation du crédit uniquement

def load_data():
    df = pd.read_csv("data/application_train.csv")
    bureau = pd.read_csv("data/bureau.csv")
    prev = pd.read_csv("data/previous_application.csv")
    install = pd.read_csv("data/installments_payments.csv")
    return df, bureau, prev, install

def engineer_features(df, bureau, prev, install):
    # 🔹 Agrégations
    bureau_agg = bureau.groupby("SK_ID_CURR").agg({
        "AMT_CREDIT_SUM": "mean",
        "CREDIT_ACTIVE": lambda x: (x == "Active").sum()
    }).rename(columns={
        "AMT_CREDIT_SUM": "BUREAU_CREDIT_MEAN",
        "CREDIT_ACTIVE": "BUREAU_ACTIVE_COUNT"
    }).reset_index()

    prev_agg = prev.groupby("SK_ID_CURR").agg({
        "AMT_APPLICATION": "mean",
        "AMT_CREDIT": "mean",
    }).rename(columns={
        "AMT_APPLICATION": "PREV_APP_MEAN",
        "AMT_CREDIT": "PREV_CREDIT_MEAN"
    }).reset_index()

    install["LATE_PAYMENT"] = (install["AMT_PAYMENT"] < install["AMT_INSTALMENT"]).astype(int)
    install_agg = install.groupby("SK_ID_CURR").agg({
        "LATE_PAYMENT": "mean"
    }).rename(columns={
        "LATE_PAYMENT": "LATE_PAYMENT_RATE"
    }).reset_index()

    # 🔹 Fusion
    df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")
    df = df.merge(prev_agg, on="SK_ID_CURR", how="left")
    df = df.merge(install_agg, on="SK_ID_CURR", how="left")

    # 🔹 Préparation
    df = df.drop(columns=["SK_ID_CURR", "SCORE"], errors='ignore')
    df.fillna(0, inplace=True)
    return df

def train_models(X, y):
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Modèles à tester
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    best_model = None
    best_auc = 0
    best_name = ""

    print("🎯 Résultats des modèles :")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        print(f" - {name} : AUC = {auc:.4f}")

        if auc > best_auc:
            best_model = model
            best_auc = auc
            best_name = name

    return best_model, scaler, best_name, best_auc

def save_model(model, scaler):
    Path("model").mkdir(exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    print("📁 Sauvegardé dans model/model.pkl + model/scaler.pkl")

if __name__ == "__main__":
    df, bureau, prev, install = load_data()
    df_ready = engineer_features(df, bureau, prev, install)

    X = df_ready.drop(columns=["TARGET"])
    y = df_ready["TARGET"]

    best_model, scaler, best_name, best_auc = train_models(X, y)
    print(f"\n✅ Modèle sélectionné : {best_name} avec AUC = {best_auc:.4f}")

    save_model(best_model, scaler)
    print(X.columns.tolist())

