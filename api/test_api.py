"""Tests de fumée de l'API (chargés en CI). Lancer depuis api/ : pytest -q"""
import numpy as np
from fastapi.testclient import TestClient
from main import app, cost_min_decision

client = TestClient(app)

VITALE = {
    "age": 67, "freq_cardiaque": 122, "tension_sys": 85, "temp": 39.1, "sat_oxygene": 86,
    "antecedents": 1, "duree_symptomes": 2, "source": "appel",
    "description_symptomes": "Douleur thoracique intense, perte de connaissance.",
}
NON_URGENT = {
    "age": 30, "freq_cardiaque": 72, "tension_sys": 120, "temp": 37.0, "sat_oxygene": 99,
    "antecedents": 0, "duree_symptomes": 48, "source": "chat",
    "description_symptomes": "Demande de certificat medical pour le sport.",
}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_valid_level():
    r = client.post("/predict", json=VITALE)
    assert r.status_code == 200
    assert r.json()["niveau_urgence"] in (0, 1, 2)


def test_validation_rejects_out_of_range():
    assert client.post("/predict", json={**NON_URGENT, "sat_oxygene": 200}).status_code == 422


# matrice de coûts métier : COST[vraie_classe, classe_prédite]
COST = [[0, 1, 2], [5, 0, 1], [20, 8, 0]]


def test_cost_min_prefers_vital_when_ambiguous():
    # proba 50/50 entre non-urgent et vital : rater une vitale (2->0) coûte 20,
    # sur-trier (0->2) coûte 2 -> la règle de coût minimal doit choisir vital (2).
    assert cost_min_decision(np.array([0.5, 0.0, 0.5]), COST) == 2


def test_cost_min_differs_from_argmax():
    # argmax choisirait 0 (proba max), mais le coût minimal préfère 2 (plus sûr).
    proba = np.array([0.5, 0.0, 0.5])
    assert int(proba.argmax()) == 0
    assert cost_min_decision(proba, COST) == 2
