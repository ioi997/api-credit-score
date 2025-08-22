import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from pathlib import Path

# Dossier cible = ./data/
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def save_path(name):
    return DATA_DIR / f"{name}.csv"

def gen_application_train(n):
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100001, 100001 + n),
        "DAYS_BIRTH": -np.random.randint(20*365, 70*365, n),
        "AMT_INCOME_TOTAL": np.random.randint(1000, 20000, n),
        "AMT_CREDIT": np.random.randint(500, 500000, n),
        "AMT_ANNUITY": np.random.randint(100, 5000, n),
        "DAYS_EMPLOYED": -np.random.randint(0, 40*365, n),
    })

    score = (
        0.15 * (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]).clip(0, 5) +
        0.10 * (df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]).clip(0, 2) +
        0.05 * (df["DAYS_BIRTH"] > -25*365).astype(int) +
        0.05 * (df["DAYS_EMPLOYED"] > -2*365).astype(int) +
        np.random.rand(n) * 0.15
    )

    df["SCORE"] = score
    df["TARGET"] = (score > 0.8).astype(int)

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(df.drop(columns="TARGET"), df["TARGET"])
    df_bal = X_res.copy()
    df_bal["TARGET"] = y_res
    df_bal["SK_ID_CURR"] = np.arange(100001, 100001 + len(df_bal))

    return df_bal

def gen_application_test(n):
    return gen_application_train(n).drop(columns=["SCORE", "TARGET"])

def gen_previous_application(n, app_df):
    return pd.DataFrame({
        "SK_ID_PREV": np.arange(400001, 400001 + n),
        "SK_ID_CURR": np.random.choice(app_df["SK_ID_CURR"], n),
        "AMT_APPLICATION": np.random.randint(500, 300000, n),
        "AMT_CREDIT": np.random.randint(500, 300000, n),
        "NAME_CONTRACT_STATUS": np.random.choice(["Approved", "Refused"], n),
    })

def gen_installments(n, app_df):
    return pd.DataFrame({
        "SK_ID_CURR": np.random.choice(app_df["SK_ID_CURR"], n),
        "AMT_INSTALMENT": np.random.randint(100, 5000, n),
        "AMT_PAYMENT": np.random.randint(50, 5000, n),
        "DAYS_INSTALMENT": -np.random.randint(1, 1500, n),
        "DAYS_ENTRY_PAYMENT": -np.random.randint(1, 1500, n),
    })

def gen_bureau(n, app_df):
    return pd.DataFrame({
        "SK_ID_BUREAU": np.arange(300001, 300001 + n),
        "SK_ID_CURR": np.random.choice(app_df["SK_ID_CURR"], n),
        "CREDIT_ACTIVE": np.random.choice(["Active", "Closed"], n),
        "AMT_CREDIT_SUM": np.random.randint(500, 200000, n),
        "DAYS_CREDIT": -np.random.randint(0, 2000, n),
    })

def generate_all():
    print("📊 Génération des datasets...")

    train = gen_application_train(5000)
    test = gen_application_test(2000)
    bureau = gen_bureau(8000, train)
    prev = gen_previous_application(6000, train)
    install = gen_installments(8000, train)

    datasets = {
        "application_train": train,
        "application_test": test,
        "bureau": bureau,
        "previous_application": prev,
        "installments_payments": install,
    }

    for name, df in datasets.items():
        df.to_csv(save_path(name), index=False)

    print("✅ Données sauvegardées dans :", DATA_DIR)

    print("\n🎯 Répartition TARGET :")
    print(train["TARGET"].value_counts(normalize=True))

    train["SCORE"].hist(bins=40)
    plt.title("Distribution des scores simulés")
    plt.xlabel("Score")
    plt.ylabel("Clients")
    plt.show()

    print("\n📐 Dimensions des datasets :")
    for name, df in datasets.items():
        print(f"- {name:25s} ➤ {df.shape}")

if __name__ == "__main__":
    generate_all()
