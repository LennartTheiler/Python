# pipeline.py
import pandas as pd
import logging
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

def extract(csv_path: str) -> pd.DataFrame:
    """Lädt eine CSV-Datei in einen DataFrame."""
    df = pd.read_csv(csv_path)
    logger.info(f"Extracted {len(df)} rows from {csv_path}")
    return df

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Entfernt Duplikate und füllt Nullwerte mit 0."""
    before = len(df)
    df_clean = df.drop_duplicates().fillna(0)
    logger.info(f"Transformed data: removed {before - len(df_clean)} duplicates")
    return df_clean

def load(df: pd.DataFrame, db_url: str, table_name: str):
    """Schreibt den DataFrame in eine Datenbank-Tabelle."""
    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    logger.info(f"Loaded {len(df)} rows into table '{table_name}' at '{db_url}'")
