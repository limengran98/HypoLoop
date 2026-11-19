"""One interface for native model bundles and both API services."""
import argparse
import json
from pathlib import Path
import numpy as np
import copy
import torch
from .loop import FeedbackEditor
from .stream import FeedbackStream
from .tasks import load_bundle
from .transport import APIConfig, APIPlanner


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    providers = [args.provider] if args.provider != "both" else ["deepseek", "claude"]
    tasks = {p: load_bundle(args.bundle, args.device) for p in providers}
    editors = {p: FeedbackEditor(t.native, APIPlanner(APIConfig.from_env(p), root/p/"api"),
                               reference_ids=t.reference_ids, metric=t.metric,
                               validate=t.validate, output=root/p/"editing")
               for p, t in tasks.items()}
    task = next(iter(tasks.values()))
    roles = task.manifest["roles"]
    sequence = list(roles)
    stream = FeedbackStream(sequence, task.targets, root/"predictions")
    known_ids, known_targets = [], []
    for index, role in enumerate(sequence):
        ids = roles[role]
        if known_ids:
            target = np.concatenate(known_targets)
            for provider, editor in editors.items():
                editor.update(known_ids, target, task.folds(known_ids))
        predictions={p:e.predict(ids).cpu().numpy() for p,e in editors.items()}
        for provider,editor in editors.items():
            checkpoint=root/provider/"checkpoints"/(role+".pt")
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            if editor.current:
                editor.current[0].save(checkpoint,editor.current[1])
            else:
                torch.save(dict(model_state_dict=editor.native.model.state_dict()),checkpoint)
            fresh=copy.deepcopy(editor.native.model)
            fresh.load_state_dict(torch.load(checkpoint,map_location=editor.native.device,
                                             weights_only=True)["model_state_dict"])
            with torch.no_grad():
                replay=editor.native.predict_module(fresh,ids).double().cpu().numpy()
            if not np.allclose(replay,predictions[provider],rtol=1e-5,atol=2e-6):
                raise ValueError("Native checkpoint replay differs before feedback release.")
            del fresh
        stream.seal(role, ids, predictions)
        target = stream.reveal(role)
        scores = {p: task.score(predictions[p], target, ids) for p in editors}
        (root/(role+"_scores.json")).write_text(json.dumps(scores, indent=2))
        known_ids.extend(ids)
        known_targets.append(target)
    print("Completed", len(sequence), "sealed feedback batches:", root)


def main():
    parser = argparse.ArgumentParser(description="HypoLoop native editing")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--provider", choices=["deepseek","claude","both"], default="both")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
