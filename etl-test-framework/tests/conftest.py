# conftest.py
import pytest
import pandas as pd
from faker import Faker
import os

fake = Faker()

@pytest.fixture
def sample_csv(tmp_path) -> str:
    """
    Erstellt eine temporäre CSV-Datei mit gefälschten Beispieldaten zur Verwendung in Tests.

    Parameter:
    tmp_path (pytest.Fixture): Von pytest bereitgestelltes temporäres Verzeichnis

    Rückgabe:
    str: Pfad zur generierten CSV-Datei als String
    """
    data = {
        "id": range(100),
        "name": [fake.name() for _ in range(100)],
        "value": [fake.random_number(digits=3) for _ in range(100)],
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "data.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)
