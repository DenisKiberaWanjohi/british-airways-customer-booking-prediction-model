from dataclasses import dataclass, asdict
from pathlib import Path
import json

SCHEMA_PATH = Path("models/feature_info.json")

@dataclass
class FeatureSchema:
    numeric: list
    categorical: list
    target: str
    class_labels: list
    n_rows: int | None = None

    def save(self, path: Path = SCHEMA_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path = SCHEMA_PATH):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
