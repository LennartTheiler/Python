# tests/test_pipeline.py
import pytest
import pandas as pd
from sqlalchemy import create_engine
from etl.pipeline import extract, transform, load

@pytest.fixture
def sample_csv(tmp_path):
    # Erstelle eine kleine CSV mit Fake‑Daten
    df = pd.DataFrame({
        "id": [1, 2, 2, 3, None],
        "name": ["Alice", "Bob", "Bob", "Charlie", "Dave"],
        "value": [10, 20, 20, 30, None],
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_extract(sample_csv):
    df = extract(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 5

def test_transform(sample_csv):
    df = extract(sample_csv)
    df2 = transform(df)
    # keine Duplikate, keine NaNs
    assert df2.duplicated().sum() == 0
    assert df2.isnull().sum().sum() == 0

def test_load(sample_csv):
    df = transform(extract(sample_csv))
    db_url = "sqlite:///:memory:"
    table = "tbl"
    load(df, db_url, table)

    engine = create_engine(db_url)
    df_db = pd.read_sql_table(table, engine)
    pd.testing.assert_frame_equal(df.reset_index(drop=True),
                                  df_db.reset_index(drop=True))
