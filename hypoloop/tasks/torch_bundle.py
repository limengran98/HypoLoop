"""Load native inputs, an original checkpoint, and declared feedback roles."""
from pathlib import Path
from types import SimpleNamespace
import importlib
import json
import numpy as np
import torch
from ..native import NativeModel
from ..metrics import mse, mae, rmse, relative_l2, abs_rel


def resolve(name):
    module, function = name.split(":", 1)
    return getattr(importlib.import_module(module), function)


def load_bundle(directory, device="cpu"):
    folder = Path(directory)
    manifest = json.loads((folder/"manifest.json").read_text())
    model = resolve(manifest["model_factory"])(**manifest.get("model_kwargs", {}))
    state = torch.load(folder/manifest["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model = model.to(device).eval()
    inputs = torch.load(folder/manifest["inputs"], map_location=device, weights_only=True)
    custom = resolve(manifest["forward"]) if manifest.get("forward") else None
    def predict(module, ids):
        output = []
        for identity in ids:
            item = inputs[identity]
            if custom:
                value = custom(module, item)
            elif isinstance(item, dict):
                value = module(**item)
            elif isinstance(item, (tuple, list)):
                value = module(*item)
            else:
                value = module(item)
            output.append(value.reshape(1, -1))
        return torch.cat(output)
    native = NativeModel(model, predict, output_scale=manifest["output_scale"],
                         binary=manifest.get("binary", False))
    metric_functions = dict(mse=mse, mae=mae, rmse=rmse,
                            relative_l2=relative_l2, abs_rel=abs_rel)
    score = resolve(manifest["metric"]) if ":" in manifest["metric"] else metric_functions[manifest["metric"]]
    def metric(prediction,target,ids=None):
        value=score(prediction,target,ids)
        return 1-value if manifest.get("higher_is_better",False) else value
    validate=resolve(manifest["validate"]) if manifest.get("validate") else None
    def targets(role, ids):
        with np.load(folder/manifest["targets"][role], allow_pickle=False) as data:
            if data["ids"].astype(str).tolist() != list(ids):
                raise ValueError("Feedback target identities differ from the input manifest.")
            return data["y"].copy()
    def folds(ids):
        key = str(len(ids))
        records = manifest["folds"][key]
        return [(np.asarray(row["train"]), np.asarray(row["validation"])) for row in records]
    return SimpleNamespace(native=native, manifest=manifest, targets=targets, folds=folds,
                           metric=metric, score=score, validate=validate, reference_ids=manifest["reference_ids"])
