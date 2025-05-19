# Dieses Programm ist die main-Datei meiner Streamlit-Webanwendung zur Erkennung von Anomalien in 
# Cloud-Logdateien. Es ermöglicht den Upload einer JSON-basierten Logdatei, extrahiert relevante 
# Textinformationen, wandelt diese mittels eines Transformer-Modells in semantische Embeddings 
# um und erkennt mithilfe eines Anomaly-Detection-Algorithmus ungewöhnliche Logeinträge. Die 
# Ergebnisse werden tabellarisch und gefiltert (nur Anomalien) visualisiert.

import streamlit as st
import pandas as pd
from preprocess import load_logs, extract_text
from embed import embed
from detect import train_detector, predict

st.title("Cloud Log Anomaly Detector")

# 1) Datei‑Upload
uploaded = st.file_uploader("Lade JSON‑Logdatei hoch", type="json")
if uploaded:
    # temporär abspeichern
    path = f"tmp_{uploaded.name}"
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

    # 2) Log‑Vorverarbeitung
    logs = load_logs(path)
    texts = extract_text(logs)
    st.write("Extrahierte Texte:")
    st.write(texts)

    # 3) Embedding
    with st.spinner("Erstelle Embeddings..."):
        embs = embed(texts)

    # 4) Anomaly Detection
    with st.spinner("Trainiere Detektor..."):
        clf = train_detector(embs)
        scores, labels = predict(clf, embs)

    # 5) Ergebnisse anzeigen
    df = pd.DataFrame({
        "text": texts,
        "score": scores,
        "anomaly": labels
    })
    st.subheader("Anomalie‑Ergebnisse")
    st.dataframe(df)

    # 6) Filter: nur Anomalien
    st.subheader("Verdächtige Einträge")
    st.table(df[df.anomaly == 1])
