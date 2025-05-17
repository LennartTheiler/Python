# conftest.py
import pytest
import pandas as pd
from faker import Faker
import os

fake = Faker()

@pytest.fixture
def sample_csv(tmp_path) -> str:
    # Erstelle ein DataFrame mit 100 Zeilen Fake-Daten
    data = {
        "id": range(100),
        "name": [fake.name() for _ in range(100)],
        "value": [fake.random_number(digits=3) for _ in range(100)],
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "data.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)
