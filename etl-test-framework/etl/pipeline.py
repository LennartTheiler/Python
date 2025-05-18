# pipeline.py
import pandas as pd
import logging
from sqlalchemy import create_engine

# Initialisierung eines Loggers für das Modul
logger = logging.getLogger(__name__)

def extract(csv_path: str) -> pd.DataFrame:
    """
    Lädt Daten aus einer CSV-Datei und gibt sie als DataFrame zurück.

    Parameter:
    csv_path (str): Pfad zur CSV-Datei

    Rückgabe:
    pd.DataFrame: Eingelesene Daten
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Extracted {len(df)} rows from {csv_path}")
    return df

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereinigt den DataFrame, indem Duplikate entfernt und fehlende Werte ersetzt werden.

    Parameter:
    df (pd.DataFrame): Ursprünglicher DataFrame

    Rückgabe:
    pd.DataFrame: Bereinigter DataFrame
    """
    before = len(df)
    df_clean = df.drop_duplicates().fillna(0)
    logger.info(f"Transformed data: removed {before - len(df_clean)} duplicates")
    return df_clean

def load(df: pd.DataFrame, db_url: str, table_name: str):
    """
    Schreibt einen DataFrame in eine SQL-Datenbanktabelle.

    Parameter:
    df (pd.DataFrame): Zu ladender DataFrame
    db_url (str): Verbindungs-URL zur Datenbank
    table_name (str): Name der Zieltabelle
    """
    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    logger.info(f"Loaded {len(df)} rows into table '{table_name}' at '{db_url}'")
