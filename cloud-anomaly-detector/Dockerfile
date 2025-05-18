FROM python:3.11-slim

# Arbeitsverzeichnis im Container
WORKDIR /app

# Requirements kopieren und installieren
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Rest des Codes kopieren
COPY src/ ./

# Port öffnen für Streamlit
EXPOSE 8501

# Startbefehl: Streamlit-App starten
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
