# main.py
from etl.pipeline import extract, transform, load
import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"

def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    # 1) Definiere Pfad zur CSV und Ziel-Datenbank
    csv_path = "sample.csv"
    db_url = "sqlite:///output.db"
    table_name = "demo_table"

    # 2) Ausführen der ETL-Schritte
    try:
        df = extract(csv_path)
        df_clean = transform(df)
        load(df_clean, db_url, table_name)
        logging.info("ETL-Prozess erfolgreich abgeschlossen.")
    except Exception as e:
        logging.exception("Fehler im ETL-Prozess:")
        sys.exit(1)

if __name__ == "__main__":
    main()
