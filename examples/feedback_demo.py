"""CPU walkthrough: returned feedback -> revised plan -> native checkpoint."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from hypoloop import NativeModel, EditPlan, FitSpec, FeedbackEditor
from hypoloop.metrics import mse, grouped_folds
from hypoloop.planners import RecordedPlanner


def main(output):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    x = torch.as_tensor(rng.uniform(-1,1,(48,2)), dtype=torch.float64)
    model = torch.nn.Sequential(torch.nn.Linear(2,4), torch.nn.Tanh(), torch.nn.Linear(4,1)).double()
    ids = [f"sample_{i:03d}" for i in range(len(x))]
    lookup = {g:i for i,g in enumerate(ids)}
    def predict(module, names):
        return module(x[[lookup[g] for g in names]])
    native = NativeModel(model, predict)
    with torch.no_grad():
        target = predict(model, ids).numpy()+0.5
    feedback, query = ids[:12], ids[12:]
    plans = [EditPlan(("2.bias",), FitSpec(ridge=0.01), "Constant positive error.", "Correct the offset."),
             EditPlan(("2.weight","2.bias"), FitSpec(loss="smooth_mae",ridge=0.01),
                      "Test the output representation and bias together.", "Improve held-out predictions.")]
    editor = FeedbackEditor(native, RecordedPlanner(plans), reference_ids=ids[12:16],
                            output=output)
    folds = grouped_folds(feedback, {g:g for g in feedback}, maximum=2)
    baseline = mse(native.predict(query).numpy(), target[12:])
    event = editor.update(feedback, target[:12], folds)
    updated = mse(editor.predict(query).numpy(), target[12:])
    checkpoint = Path(output)/"round_01/model.pt"
    reloaded = torch.nn.Sequential(torch.nn.Linear(2,4), torch.nn.Tanh(), torch.nn.Linear(4,1)).double()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True)["model_state_dict"])
    with torch.no_grad():
        replay = predict(reloaded, query).numpy()
    error = float(np.max(np.abs(replay-editor.predict(query).numpy())))
    report = dict(example="synthetic native-editing walkthrough", base_mse=baseline,
                  updated_mse=updated, native_reload_max_error=error,
                  candidates=len(event["candidates"]), selected=event["selected"])
    (Path(output)/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="selected"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/demo")
    main(parser.parse_args().output)
