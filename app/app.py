"""Interface de tri d'urgence — saisie d'une situation, prédiction, historique."""
import os
from datetime import datetime

import requests
import streamlit as st
from loguru import logger

API_URL = os.getenv("API_URL", f"http://localhost:{os.getenv('FASTAPI_PORT', '8080')}").rstrip("/")
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="INFO")

NIVEAU_STYLE = {
    0: ("🟢 Non urgent", "#16a34a"),
    1: ("🟠 Urgence relative", "#ca8a04"),
    2: ("🔴 Urgence vitale", "#dc2626"),
}

st.set_page_config(page_title="Tri d'urgence télémédecine", page_icon="🚑")
st.title("Tri d'urgence — télémédecine")
st.caption("Saisissez la situation : le modèle estime le niveau d'urgence (0/1/2).")

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("triage"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Âge", 0, 120, 50)
        freq_cardiaque = st.number_input("Fréquence cardiaque (bpm)", 20.0, 300.0, 80.0)
        tension_sys = st.number_input("Tension systolique (mmHg)", 40.0, 300.0, 120.0)
        temp = st.number_input("Température (°C)", 30.0, 45.0, 37.0)
        sat_oxygene = st.number_input("Saturation O₂ (%)", 50.0, 100.0, 98.0)
    with c2:
        duree_symptomes = st.number_input("Durée des symptômes (h)", 0.0, 1000.0, 12.0)
        antecedents = st.selectbox("Pathologie chronique ?", [("Non", 0), ("Oui", 1)], format_func=lambda x: x[0])[1]
        source = st.selectbox("Source", ["appel", "chat"])
    description_symptomes = st.text_area("Description des symptômes", "Douleur thoracique et essoufflement.")
    submitted = st.form_submit_button("Évaluer l'urgence")

if submitted:
    payload = {
        "age": int(age), "freq_cardiaque": freq_cardiaque, "tension_sys": tension_sys,
        "temp": temp, "sat_oxygene": sat_oxygene, "antecedents": int(antecedents),
        "duree_symptomes": duree_symptomes, "source": source,
        "description_symptomes": description_symptomes,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        niveau = result["niveau_urgence"]
        label, color = NIVEAU_STYLE[niveau]
        st.markdown(
            f"<div style='padding:16px;border-radius:8px;background:{color}22;"
            f"border:2px solid {color}'><h3 style='color:{color};margin:0'>{label}</h3></div>",
            unsafe_allow_html=True,
        )
        st.write("**Probabilités :**")
        st.bar_chart(result["probabilities"])
        logger.info(f"prediction niveau={niveau} session={result['session_id']}")
        st.session_state.history.insert(0, {
            "heure": datetime.now().strftime("%H:%M:%S"),
            "niveau": niveau, "label": label,
            "description": description_symptomes[:50],
        })
    except requests.exceptions.RequestException as exc:
        st.error(f"Erreur API : {exc}")
        logger.error(f"Erreur API : {exc}")

if st.session_state.history:
    st.subheader("Historique des inférences")
    st.dataframe(st.session_state.history, width="stretch")
