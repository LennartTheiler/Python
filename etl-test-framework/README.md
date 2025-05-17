# ETL Test Framework

Ein leichtgewichtiges Python-basiertes Integrations-Test-Framework für ETL-Pipelines (CSV → Pandas → Datenbank), 
das automatisch Testdaten generiert, Transformationen validiert und Ergebnisse in einer Datenbank lädt.

## Features

- **Extract**: Liest CSV-Dateien mittels Pandas  
- **Transform**: Entfernt Duplikate und füllt Nullwerte auf  
- **Load**: Schreibt Daten in eine SQLite-Datenbank (oder über SQLAlchemy in andere DBs)  
- **Automatisierte Tests**: pytest-Tests mit dynamischen Fixtures (Faker)  
- **Reporting**: HTML-Reports (pytest-html) und Coverage-Reports (pytest-cov)  
- **Log-Ausgabe**: Detaillierte Log-Messages während der Ausführung  

## Projektstruktur

etl-test-framework/
├── etl/ # Dein ETL-Paket
│ ├── init.py # Package-Marker
│ └── pipeline.py # Extract, Transform, Load
├── tests/ # pytest Tests
│ ├── init.py
│ └── test_pipeline.py # Unit- & Integrationstests
├── sample.csv # Beispiel-Datei
├── main.py # Hauptskript
├── pytest.ini # pytest-Konfiguration
├── requirements.txt # Abhängigkeiten
└── README.md # Projektdokumentation
