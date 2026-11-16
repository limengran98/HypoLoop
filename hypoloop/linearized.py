"""Linearized proposals through selected coordinates of the original model."""
from dataclasses import asdict
import time
import numpy as np
import torch
from .linear import fit
from .solver import FitResult


def fit_linearized(space, ids, target, reference_ids, spec, *, ages=None, steps=80):
    spec.validate(space.native.binary)
    started = time.monotonic()
    w0 = space.w0.detach().requires_grad_(True)
    p0 = space.predict(ids, w0).cpu().numpy()
    jac = torch.autograd.functional.jacobian(lambda w: space.forward(ids, w), w0).detach().cpu().numpy()
    if reference_ids:
        ref = torch.autograd.functional.jacobian(lambda w: space.forward(reference_ids, w), w0).detach().cpu().numpy()
    else:
        if spec.preservation:
            raise ValueError("Prediction preservation requires reference inputs.")
        ref = np.zeros((1,p0.shape[1],space.dimension))
    values, detail = fit(p0, jac, space.w0.cpu().numpy(), space.scales.cpu().numpy(),
                         np.asarray(target), ref, np.arange(space.dimension), asdict(spec),
                         output_scale=space.native.output_scale, age=ages,
                         max_iterations=steps)
    values = space.pack(torch.as_tensor(values, device=space.device, dtype=torch.float64))
    space.native.check_original()
    return FitResult(values, float(detail["objective_after_native_rounding"]), time.monotonic()-started,
                     int(detail["cost"]["optimizer_steps"]),
                     int(torch.count_nonzero(values != space.w0)), int(np.isfinite(target).sum()))
