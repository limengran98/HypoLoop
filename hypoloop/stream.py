"""Seal predictions before releasing the next feedback batch."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np


class FeedbackStream:
    def __init__(self, roles, target_loader, directory):
        self.roles, self.target_loader = list(roles), target_loader
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.position = 0

    def seal(self, role, ids, predictions):
        if self.position >= len(self.roles) or role != self.roles[self.position]:
            raise ValueError("Feedback batches must follow the declared order.")
        path = self.directory/(role+".npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, ids=np.asarray(ids),
                                **{k: np.asarray(v) for k, v in predictions.items()})
        record = dict(role=role, ids=list(ids), predictors=sorted(predictions),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      sealed_at=datetime.now(timezone.utc).isoformat())
        (self.directory/(role+".seal.json")).write_text(json.dumps(record, indent=2))
        return record

    def reveal(self, role):
        if self.position >= len(self.roles) or role != self.roles[self.position]:
            raise ValueError("A feedback batch cannot be skipped or released twice.")
        path = self.directory/(role+".npz")
        record = json.loads((self.directory/(role+".seal.json")).read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("The prediction seal differs.")
        target = self.target_loader(role, record["ids"])
        (self.directory/(role+".released.json")).write_text(json.dumps(
            dict(role=role, prediction_sha256=record["sha256"],
                 released_at=datetime.now(timezone.utc).isoformat()), indent=2))
        self.position += 1
        return target
