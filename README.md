# Diagnostic assisté et tri d'urgence multimodal en télémédecine

## tl;dr:

### Installation locale (sans Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Lancer le notebook
jupyter lab notebooks/analyse.ipynb

# Lancer MLflow UI
mlflow ui

# Lancer l'API
uvicorn api.main:app --reload

# Lancer le frontend
streamlit run frontend/app.py
```

### Avec Docker

```bash
docker compose up --build
```

- API : http://localhost:8000/docs (Swagger)
- Frontend : http://localhost:8501
- MLflow : http://localhost:5000
- Prometheus : http://localhost:9090
- Grafana : http://localhost:3000 (admin / admin) — dashboard 1860 préchargé
- node-exporter : http://localhost:9100/metrics

## Étapes du projet

1. **EDA + nettoyage** (notebook) — qualité données, NaN, outliers, RGPD.
2. **Modélisation** (notebook) — 3 modèles × 4 scénarios, validation croisée, MLflow tracking.
3. **Choix du modèle** — meilleur compromis performance / latence + ajustement pour minimiser les faux négatifs sur la classe "urgence vitale".
4. **Industrialisation** — API FastAPI + Streamlit + Docker.
5. **CI/CD** — GitHub Actions : lint, tests, build images, push GHCR.
6. **Éthique & RGPD** — section dédiée dans le notebook et le README.

