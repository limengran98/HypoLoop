# Load the HypoLoop model and data assets

The model repository contains selected native checkpoints, named-tensor replacements, and registered feedback data. Use the bundle's `manifest.json` for task IDs, exact filenames, source checkpoints, and variant names. [Hugging Face model repository](https://huggingface.co/Amanda1998/HypoLoop)

```bash
pip install torch numpy huggingface_hub safetensors
hf download Amanda1998/HypoLoop --local-dir ./hypoloop-assets
```

## Checkpoints

Use `load_state` to recover the original PyTorch state-dict tensor names and types. Fourier weights use trailing real/imaginary pairs on disk.

```python
import json

import torch
from safetensors import safe_open


def load_state(path):
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        state = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
        complex_keys = json.loads(checkpoint.metadata()["complex_keys"])
    for name in complex_keys:
        state[name] = torch.view_as_complex(state[name])
    return state
```

- **Burgers.** Instantiate the pinned PDEBench `FNO1d` with the arguments in `models/burgers/architecture.json`. Load a full state dict with `strict=True` and run the model with the `x` histories and expanded spatial `grid` from `data/burgers/inputs.npz`. [PDEBench FNO source](https://github.com/pdebench/PDEBench/blob/4ff3e3a4aa1561721b5571fa3a048a0a463e0568/pdebench/models/fno/fno.py)
- **Constrained QP.** Use DC3's `NNSolver` and `SimpleProblem` with the saved `model_kwargs`, the objective/constraint matrices and the completion blocks in `data/constrained-qp/quadratic_program.npz`. Pass the saved right-hand sides through `NNSolver` and the original `grad_steps_all` correction. [DC3 upstream](https://github.com/locuslab/DC3)
- **DIODE and Make3D.** Instantiate the metric VKITTI ViT-S model at `max_depth=80` as directed upstream. Verify the original checkpoint SHA-256 in the architecture JSON, load the upstream state dict, then replace the listed named tensors with one SafeTensors file. Load the merged state with `strict=True`. Each replacement stores final tensor values. [Depth Anything V2 metric model](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small)

### Reconstruct the registered DC3 problem

Install the upstream DC3 dependencies and make its modules available on the Python path. Reuse the saved completion coordinates and inverse when instantiating the problem to reproduce the original numerical path.

```python
from pathlib import Path

import numpy as np
from method import NNSolver, grad_steps_all
from utils import SimpleProblem

root = Path("hypoloop-assets")
config = json.loads((root / "models/constrained-qp/architecture.json").read_text())
args = config["model_kwargs"]
with np.load(root / "data/constrained-qp/quadratic_program.npz", allow_pickle=False) as arrays:
    problem = SimpleProblem.__new__(SimpleProblem)
    for name in arrays.files:
        value = arrays[name]
        setattr(problem, "_" + name, value if name.endswith("_vars") else torch.from_numpy(value))
problem._xdim = problem._neq = problem.A.shape[0]
problem._nineq = problem.G.shape[0]
problem._ydim = problem.Q.shape[0]
problem._nknowns = 0
problem._device = torch.device("cpu")

model = NNSolver(problem, args).double().eval()
model.load_state_dict(load_state(root / "models/constrained-qp/deepseek-dynamic-location.safetensors"))
with np.load(root / "data/constrained-qp/inputs.npz", allow_pickle=False) as arrays:
    x = torch.from_numpy(arrays["x"][:1])
with torch.inference_mode():
    y, _ = grad_steps_all(problem, x, model(x), args)
```

## Feedback data

Read NPZ files with `allow_pickle=False` and join inputs and targets using `ids`. Read feedback in order `F0`, `B1`, `B2`, then use `E` for evaluation. In DIODE, `images.json` maps each ID to an RGB image and scene. Gather predictions at the original-image rows and columns in `grid.json` and ignore NaN targets. Original datasets for all 22 paper tasks are described in the catalog. [Task and model catalog](DATASETS.md)
