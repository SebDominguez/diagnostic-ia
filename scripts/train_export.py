"""Entraîne le modèle de production et l'exporte pour l'API.

Modèle retenu (cf. notebook, sections 16-21) :
  - LightGBM sur le Scénario 2 (SANS variables sensibles sexe/zone_vie) → conforme RGPD
  - Décision par COÛT MINIMAL (matrice de coûts métier) appliquée côté API

Sortie : api/model/urgence_model.joblib  (Pipeline sklearn complet + matrice de coûts + métadonnées)

Ce script est appelable en CI/CD (étape "entraînement du modèle").
Usage : python scripts/train_export.py
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os

import numpy as np
import pandas as pd
import joblib
import sklearn
import lightgbm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, confusion_matrix, balanced_accuracy_score
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
# Chemins surchargeables par variables d'env (utile en conteneur, cf. compose.yml).
# Défaut local inchangé : data/processed/... et api/model/...
PROCESSED = Path(os.getenv("DATA_PROCESSED", ROOT / "data" / "processed" / "dataset_telemed_processed.csv"))
OUT_DIR = Path(os.getenv("MODEL_OUT_DIR", ROOT / "api" / "model"))
OUT_PATH = OUT_DIR / "urgence_model.joblib"

# --- Variables d'entrée du modèle de PRODUCTION (Scénario 2 : sans sexe ni zone_vie) ---
NUM_STD = ["age", "freq_cardiaque", "tension_sys", "temp", "duree_symptomes"]
COL_MMS = ["sat_oxygene"]
COL_BIN = ["antecedents"]
COL_CAT = ["source"]
COL_TXT = "description_symptomes"
INPUT_FEATURES = NUM_STD + COL_MMS + COL_BIN + COL_CAT + [COL_TXT]
TARGET = "niveau_urgence"
CLASS_NAMES = ["non urgent", "urgence relative", "urgence vitale"]

# Matrice de coûts métier : COST[vraie_classe, classe_prédite]
# triangle bas = sous-triage (dangereux) ; haut = sur-triage (bénin)
COST_MATRIX = [
    [0, 1, 2],
    [5, 0, 1],
    [20, 8, 0],
]


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_STD),
            ("sat", MinMaxScaler(), COL_MMS),
            ("bin", "passthrough", COL_BIN),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), COL_CAT),
            ("txt", TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=2), COL_TXT),
        ],
        remainder="drop",
    )
    # hyperparamètres alignés sur le notebook (sections 16-20), sans class_weight :
    # la décision par coût minimal a besoin de probabilités non distordues.
    clf = LGBMClassifier(
        n_estimators=200, num_leaves=15, learning_rate=0.05,
        min_child_samples=60, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="multiclass", num_class=3,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def cost_min_predict(proba: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Décision qui minimise le coût attendu : argmin_k sum_j P(j) * Cost(j, k)."""
    return (proba @ cost).argmin(axis=1)


def main() -> None:
    df = pd.read_csv(PROCESSED)
    X = df[INPUT_FEATURES].copy()
    y = df[TARGET].astype(int).values
    cost = np.array(COST_MATRIX)

    # Split pour des métriques honnêtes
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)
    y_argmax = proba.argmax(axis=1)
    y_cost = cost_min_predict(proba, cost)
    cm = confusion_matrix(y_te, y_cost)
    metrics = {
        "f1_weighted_argmax": round(f1_score(y_te, y_argmax, average="weighted"), 4),
        "f1_weighted_cost": round(f1_score(y_te, y_cost, average="weighted"), 4),
        "recall_classe2_cost": round(recall_score(y_te, y_cost, average=None)[2], 4),
        "macro_recall_cost": round(balanced_accuracy_score(y_te, y_cost), 4),
        "vitales_ratees_cost": int(cm[2, 0] + cm[2, 1]),
        "erreur_mortelle_2vers0_cost": int(cm[2, 0]),
        "cout_total_test_cost": int(sum(cost[t, p] for t, p in zip(y_te, y_cost))),
    }
    print("Métriques (test, décision coût minimal) :")
    for k, v in metrics.items():
        print(f"  {k:32} : {v}")

    # Réentraînement final sur TOUTES les données pour l'artefact de production
    pipe_full = build_pipeline()
    pipe_full.fit(X, y)

    artifact = {
        "pipeline": pipe_full,
        "cost_matrix": cost.tolist(),
        "classes": [0, 1, 2],
        "class_names": CLASS_NAMES,
        "input_features": INPUT_FEATURES,
        "scenario": "S2_sans_variables_sensibles",
        "decision_rule": "cost_minimal",
        "metrics_holdout": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lightgbm.__version__,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUT_PATH)
    (OUT_DIR / "model_metadata.json").write_text(
        json.dumps({k: v for k, v in artifact.items() if k != "pipeline"}, indent=2, ensure_ascii=False)
    )
    print(f"\nArtefact sauvegardé : {OUT_PATH}")
    print(f"Taille : {OUT_PATH.stat().st_size / 1024:.0f} Ko")


if __name__ == "__main__":
    main()
