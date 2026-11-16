# Connect a scientific predictor

HypoLoop operates on the original PyTorch model and a native serving function.
The serving function returns an object-by-output tensor and includes the
model's preprocessing, nonlinearities, and task output transformation.

~~~python
from hypoloop import NativeModel, FeedbackEditor
from hypoloop.transport import APIConfig, APIPlanner

native = NativeModel(model, predict_native, output_scale=task_scale)
planner = APIPlanner(APIConfig.from_env("deepseek"), "outputs/api")
editor = FeedbackEditor(native, planner, reference_ids=reference_ids,
                        metric=task_loss, validate=task_feasibility)
editor.update(feedback_ids, feedback_targets, registered_folds)
prediction = editor.predict(next_batch_ids)
~~~

For graphs, fields, waveforms, and images, <code>predict_native(model, ids)</code>
performs the original native request for each object. The task metric receives
the resulting predictions, targets, and identities. It defines the appropriate
group aggregation, output units, and masking.

## Native bundle interface

The command-line runner accepts a directory containing:

~~~text
scientific_task/
├── manifest.json
├── model.pt
├── inputs.pt
└── targets/
    ├── F0.npz
    ├── B1.npz
    ├── B2.npz
    └── E.npz
~~~

<code>model.pt</code> contains the original model state dictionary.
<code>inputs.pt</code> maps object identifiers to native input tensors, argument
tuples, or keyword dictionaries. Each target archive contains aligned
<code>ids</code> and <code>y</code> arrays.

The manifest specifies the model factory, native forward callback, output
scale, reference inputs, ordered feedback roles, target files, task metric,
and frozen fitting/validation indices for each feedback prefix:

~~~json
{
  "model_factory": "scientific_model:make_model",
  "model_kwargs": {},
  "checkpoint": "model.pt",
  "inputs": "inputs.pt",
  "forward": "scientific_model:forward_native",
  "metric": "scientific_model:task_loss",
  "output_scale": 1.0,
  "binary": false,
  "higher_is_better": false,
  "reference_ids": ["reference_a", "reference_b"],
  "roles": {"F0": ["f0_a", "f0_b"], "E": ["query_a"]},
  "targets": {"F0": "targets/F0.npz", "E": "targets/E.npz"},
  "folds": {"2": [
    {"train": [0], "validation": [1]},
    {"train": [1], "validation": [0]}
  ]}
}
~~~

Factories instantiate the original backbone. The native forward callback
retains canonical batching, normalization, interpolation, masking and output
decoding. For constrained optimization it also retains completion/correction.
Configure task feasibility checks through the Python <code>FeedbackEditor</code>
interface or a manifest <code>validate</code> callback. For AP/F1, set
<code>higher_is_better</code> to true: validation minimizes one minus the score,
and the runner reports the original metric. 

## Run both services

Configure the service variables in .env.example, export them
in your shell, and run:

~~~bash
python -m hypoloop.cli --bundle scientific_task --provider both --output outputs/task
~~~

DeepSeek and Claude receive the same declared model information and feedback
access. The runner seals both services' predictions before reading the target
file for the next role. Request records retain the requested and returned
model identifiers, raw replies, and executable plans. The optional
<code>forward</code> callback may be omitted for models whose standard forward
already returns the required tensor.
