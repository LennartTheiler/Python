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

```text
etl-test-framework/
├── etl/
│   ├── __init__.py
│   └── pipeline.py
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
├── sample.csv
├── main.py
├── pytest.ini
├── requirements.txt
└── README.md

