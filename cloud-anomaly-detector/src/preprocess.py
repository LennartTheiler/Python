import json
from typing import List, Dict

def load_logs(path: str) -> List[Dict]:
    """Liest CloudTrail-JSON-Zeilen aus einer Datei."""
    logs = []
    with open(path, 'r') as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return logs

def extract_text(logs: List[Dict]) -> List[str]:
    """
    Extrahiert für jeden Log-Eintrag eine Kurzbeschreibung
    z.B. "ec2.amazonaws.com:StartInstances".
    """
    texts = []
    for entry in logs:
        source = entry.get('eventSource', 'unknown')
        name   = entry.get('eventName', 'UnknownEvent')
        texts.append(f"{source}:{name}")
    return texts
