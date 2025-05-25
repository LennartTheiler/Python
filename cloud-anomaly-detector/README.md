# Cloud Anomaly Detector

Ein leistungsfähiges, dennoch einfach gehaltenes Python-basiertes Tool zur automatisierten Erkennung von Anomalien in Cloud-Logdaten. 
Das System nutzt moderne Natural Language Processing (NLP)-Techniken, indem es Logeinträge in semantische Vektor-Repräsentationen 
(Embeddings) umwandelt und mithilfe von unüberwachten (unsupervised) Machine-Learning-Algorithmen verdächtige oder abweichende Muster 
identifiziert. Dadurch ermöglicht es eine effektive Überwachung und Analyse großer Logmengen, um Sicherheitsvorfälle, 
Fehlkonfigurationen oder Betriebsstörungen frühzeitig zu erkennen und schnell darauf reagieren zu können.

Starte das Programm in einer PowerShell mit diesem Befehl:
```text
streamlit run streamlit_app.py --server.port=8501 --server.address=localhost
```

Alternativ kann das Projekt auch in einem DOcker-Container mit folgenden Befehlen laufen:
Verwende erst:
```text
docker build -t cloud-anomaly-detector .
```
und dann:
```text
docker run -p 8501:8501 cloud-anomaly-detector
```

## Features

- **Log-Upload**: Lädt CloudTrail-JSON-Logs per Webinterface hoch  
- **Vorverarbeitung**: Extrahiert relevante Kurzbeschreibungen aus den Logs  
- **Embedding**: Erzeugt semantische Vektor-Repräsentationen der Log-Texte mit Sentence Transformers  
- **Anomaly Detection**: Trainiert einen Klassifikator zur Erkennung verdächtiger Logeinträge  
- **Visualisierung**: Zeigt Anomalie-Scores und gefilterte Ergebnisse übersichtlich an  

## Projektstruktur

```text
cloud-anomaly-detector/
├── src/
|   ├── requirements.txt
│   ├── embed.py
│   ├── detect.py
│   ├── preprocess.py
│   └── streamlit_app.py
├── logs.jsonl
├── Dockerfile
└── README.md
