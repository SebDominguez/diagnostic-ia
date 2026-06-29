"""API de tri d'urgence en télémédecine.

Routes :
  GET  /          -> description du service
  POST /predict   -> prédit le niveau d'urgence (décision par coût minimal)
  POST /retrain   -> réentraînement monitoré du modèle
  GET  /health    -> santé de l'API + état du modèle
  GET  /metrics   -> métriques Prometheus

Chaque prédiction est journalisée (entrée, sortie, date, session) dans logs/predictions.jsonl.
"""
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal
import json
import subprocess
import sys
import uuid

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger
from starlette.responses import Response

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "urgence_model.joblib"
TRAIN_SCRIPT = ROOT.parent / "scripts" / "train_export.py"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
PRED_LOG = LOG_DIR / "predictions.jsonl"

app = FastAPI(title="API Tri d'urgence télémédecine", version="1.0")

# --- Métriques Prometheus ---
PRED_COUNTER = Counter("predictions_total", "Nombre de prédictions", ["niveau"])
PRED_LATENCY = Histogram("prediction_latency_seconds", "Latence de prédiction (s)")
ERROR_COUNTER = Counter("prediction_errors_total", "Erreurs de prédiction")

# --- Chargement paresseux du modèle ---
_artifact = None


def get_artifact():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Modèle non disponible ({MODEL_PATH.name}). Lancer scripts/train_export.py.",
            )
        _artifact = joblib.load(MODEL_PATH)
        logger.info(f"Modèle chargé depuis {MODEL_PATH}")
    return _artifact


def cost_min_decision(proba: np.ndarray, cost) -> int:
    """Classe qui minimise le coût attendu : argmin_k sum_j P(j) * Cost(j, k)."""
    return int((proba @ np.array(cost)).argmin())


# --- Schémas (validation des données entrantes) ---
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Âge en années")
    freq_cardiaque: float = Field(..., ge=20, le=300, description="Fréquence cardiaque (bpm)")
    tension_sys: float = Field(..., ge=40, le=300, description="Tension systolique (mmHg)")
    temp: float = Field(..., ge=30, le=45, description="Température (°C)")
    sat_oxygene: float = Field(..., ge=50, le=100, description="Saturation O2 (%)")
    antecedents: int = Field(..., ge=0, le=1, description="Pathologie chronique 1/0")
    duree_symptomes: float = Field(..., ge=0, le=1000, description="Durée des symptômes (h)")
    source: Literal["appel", "chat"] = Field(..., description="appel | chat")
    description_symptomes: str = Field(..., min_length=1, description="Récit clinique libre")
    session_id: str | None = Field(None, description="Identifiant de session (optionnel)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 67, "freq_cardiaque": 122, "tension_sys": 90, "temp": 38.9,
                "sat_oxygene": 88, "antecedents": 1, "duree_symptomes": 2,
                "source": "appel",
                "description_symptomes": "Douleur thoracique intense avec essoufflement.",
            }
        }
    }


class PredictionOutput(BaseModel):
    niveau_urgence: int
    label: str
    probabilities: dict
    decision_rule: str
    session_id: str


@app.get("/")
async def root():
    return {
        "service": "Tri d'urgence multimodal en télémédecine",
        "routes": ["/predict", "/retrain", "/health", "/metrics", "/docs"],
        "niveaux": {0: "non urgent", 1: "urgence relative", 2: "urgence vitale"},
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(patient: PatientInput):
    art = get_artifact()
    pipe = art["pipeline"]
    cost = art["cost_matrix"]
    names = art["class_names"]
    features = art["input_features"]

    row = patient.model_dump()
    session_id = row.get("session_id") or str(uuid.uuid4())
    X = pd.DataFrame([{k: row[k] for k in features}])

    try:
        with PRED_LATENCY.time():
            proba = pipe.predict_proba(X)[0]
            niveau = cost_min_decision(proba, cost)
    except Exception as exc:  # robustesse : on ne laisse jamais l'API crasher
        ERROR_COUNTER.inc()
        logger.exception("Erreur lors de la prédiction")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {exc}")

    PRED_COUNTER.labels(niveau=str(niveau)).inc()
    probabilities = {names[i]: round(float(proba[i]), 4) for i in range(len(names))}

    # Journalisation de la requête (entrée, sortie, date, session)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "input": {k: row[k] for k in features},
        "niveau_urgence": niveau,
        "label": names[niveau],
        "probabilities": probabilities,
    }
    with PRED_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"predict session={session_id} niveau={niveau} ({names[niveau]})")

    return PredictionOutput(
        niveau_urgence=niveau,
        label=names[niveau],
        probabilities=probabilities,
        decision_rule=art.get("decision_rule", "cost_minimal"),
        session_id=session_id,
    )


@app.post("/retrain")
async def retrain():
    """Réentraînement monitoré : relance le script d'entraînement et recharge le modèle."""
    if not TRAIN_SCRIPT.exists():
        raise HTTPException(status_code=503, detail="Script d'entraînement indisponible dans cet environnement.")
    start = datetime.now(timezone.utc)
    logger.info("Réentraînement déclenché")
    proc = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        logger.error(f"Échec réentraînement : {proc.stderr[-500:]}")
        raise HTTPException(status_code=500, detail="Échec du réentraînement (voir logs).")

    global _artifact
    _artifact = None  # force le rechargement du nouveau modèle
    art = get_artifact()
    duration = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(f"Réentraînement OK en {duration:.1f}s")
    return {
        "status": "ok",
        "trained_at": art.get("trained_at"),
        "metrics_holdout": art.get("metrics_holdout"),
        "duration_seconds": round(duration, 1),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
