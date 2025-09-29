from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
from typing import Dict, List

SCHEMA_PATH = Path("models/feature_info.json")

@dataclass
class FeatureSchema:
    numeric: list
    categorical: list
    target: str
    class_labels: list
    n_rows: int | None = None
    dtypes: Dict[str, str] = field(default_factory=dict)
    descriptions: Dict[str, str] = field(default_factory=dict)
    categorical_options: Dict[str, List[str]] = field(default_factory=dict)

    def save(self, path: Path = SCHEMA_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path = SCHEMA_PATH):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)