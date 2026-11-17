# Supplementary code

Requirements: Python 3.10 or later, NumPy, SciPy, and PyTorch.

Provided solely for confidential peer review. The terms are in LICENSE.

## Install

```sh
python -m pip install -e .
```

## CPU example

```sh
python examples/feedback_demo.py --output outputs/demo
```

## Tests

```sh
python -m unittest discover -s tests -v
```

## Use with a model

The code edits native PyTorch parameters. The model factory and native forward
function are supplied through a task bundle. The bundle format is described in
docs/USAGE.md. Service configuration is provided in .env.example; keys are set
through environment variables.

```sh
python -m hypoloop.cli --bundle scientific_task --provider both --output outputs/task
```

hypoloop/ contains the implementation; configs/ contains the editing recipe
and method definitions. The example runs on CPU without API access.
