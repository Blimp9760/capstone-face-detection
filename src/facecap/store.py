"""Local biometric enrollment storage. Do not sync or commit this file."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
from .pipeline import unit, similarity
import numpy as np


class EnrollmentStore:
    def __init__(self, path: Path = Path("local/enrollments.json")):
        self.path = Path(path)
        self._entries = self._load()

    def _load(self) -> dict[str, list[list[float]]]:
        if not self.path.is_file():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError("Unsupported enrollment store schema")
        return raw["identities"]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".enrollment-", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump({"schema_version": 1, "identities": self._entries}, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
            if os.name != "nt":
                self.path.chmod(0o600)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def enroll(self, identity: str, embedding: np.ndarray) -> None:
        if not identity.strip() or len(identity) > 100:
            raise ValueError("Identity must be a nonempty name of at most 100 characters")
        self._entries.setdefault(identity.strip(), []).append(unit(embedding).tolist())
        self._save()

    def delete(self, identity: str) -> bool:
        found = identity in self._entries
        if found:
            del self._entries[identity]
            self._save()
        return found

    def list_names(self) -> list[str]:
        return sorted(self._entries)

    def recognize(self, embedding: np.ndarray, threshold: float) -> tuple[str, float]:
        if not self._entries:
            return "Unknown", float("nan")
        best_name, best = "Unknown", -1.0
        for name, examples in self._entries.items():
            if examples:
                score = max(similarity(embedding, np.asarray(e)) for e in examples)
                if score > best:
                    best_name, best = name, score
        return (best_name if best >= threshold else "Unknown"), best
