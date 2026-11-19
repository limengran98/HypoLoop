"""True-native, W0-anchored optimization for an executable editing plan."""
from dataclasses import dataclass
import time
import numpy as np
import torch
from .programs import run_program


@dataclass
class FitResult:
    values: torch.Tensor
    objective: float
    seconds: float
    steps: int
    changed_coordinates: int
    observed_outputs: int


def normalized_weights(p0, target, scale, spec, ages=None):
    p0, target = np.asarray(p0), np.asarray(target, dtype=np.float64)
    observed = np.isfinite(target)
    if target.shape != p0.shape or target.ndim != 2 or not observed.any():
        raise ValueError("Feedback must contain valid object-by-output observations.")
    target = np.where(observed, target, 0.0)
    binary = spec.loss in ("bce", "brier")
    scale = 1.0 if binary else float(scale)
    prediction = 1 / (1 + np.exp(-np.clip(p0, -700, 700))) if binary else p0 / scale
    values = target if binary else target / scale
    if binary and ((target[observed] < 0).any() or (target[observed] > 1).any()):
        raise ValueError("Binary observations must lie in [0,1].")
    order = np.arange(len(target))
    age = (order.max()-order)/max(len(order)-1,1) if ages is None else np.asarray(ages)
    variables = dict(residual=prediction-values, prediction=prediction, target=values,
                     age=np.broadcast_to(age[:, None], target.shape),
                     condition=np.zeros(target.shape))
    weight = run_program(spec.weight_program, variables, target.shape) * observed
    return target / scale, weight / weight.sum(), scale, observed


def _loss(prediction, target, name):
    error = prediction-target
    if name == "bce":
        return torch.nn.functional.softplus(prediction)-target*prediction
    if name == "brier":
        return (prediction.sigmoid()-target).square()
    if name == "smooth_mae":
        return torch.hypot(error, torch.full_like(error, 0.01))-0.01
    if name == "huber":
        return torch.where(error.abs() <= 1, 0.5*error.square(), error.abs()-0.5)
    return error.square()


def fit_patch(space, ids, target, reference_ids, spec, *, ages=None, steps=80):
    """Fit a total patch; losses and weights use this fitting fold only."""
    spec.validate(space.native.binary)
    started = time.monotonic()
    ids, reference_ids = list(ids), list(reference_ids)
    p0 = space.predict(ids, space.w0).cpu().numpy()
    y, q, scale, observed = normalized_weights(p0, target, space.native.output_scale, spec, ages)
    y = torch.as_tensor(y, dtype=torch.float64, device=space.device)
    q = torch.as_tensor(q, dtype=torch.float64, device=space.device)
    reference = space.predict(reference_ids, space.w0)/scale if spec.preservation else None
    if reference is not None and not reference.numel():
        raise ValueError("A preservation objective requires reference inputs.")

    def objective(z, gradient=True):
        grad = 2*spec.ridge*z.detach()/space.dimension
        value = float(spec.ridge*z.detach().square().mean())
        for i, identity in enumerate(ids):
            if not observed[i].any():
                continue
            pred = space.forward([identity], space.w0+space.scales*z).double()/scale
            term = (q[i:i+1]*_loss(pred, y[i:i+1], spec.loss)).sum()
            value += float(term.detach())
            if gradient:
                grad = grad+torch.autograd.grad(term, z)[0].detach()
        if reference is not None:
            for i, identity in enumerate(reference_ids):
                pred = space.forward([identity], space.w0+space.scales*z).double()/scale
                term = spec.preservation*(pred-reference[i:i+1]).square().sum()/reference.numel()
                value += float(term.detach())
                if gradient:
                    grad = grad+torch.autograd.grad(term, z)[0].detach()
        if not np.isfinite(value) or not torch.isfinite(grad).all():
            raise FloatingPointError("Nonfinite native objective or gradient.")
        return value, grad

    z = torch.zeros_like(space.w0, requires_grad=True)
    first, second = torch.zeros_like(z), torch.zeros_like(z)
    best, best_value = z.detach().clone(), float("inf")
    for step in range(1, steps+1):
        value, gradient = objective(z)
        if value < best_value:
            best_value, best = value, z.detach().clone()
        with torch.no_grad():
            first, second = 0.9*first+0.1*gradient, 0.999*second+0.001*gradient.square()
            z -= 0.03*(first/(1-0.9**step))/(torch.sqrt(second/(1-0.999**step))+1e-8)
    value, _ = objective(z, gradient=False)
    if value < best_value:
        best_value, best = value, z.detach().clone()
    values = space.pack(space.w0+space.scales*best*spec.step)
    space.native.check_original()
    return FitResult(values, best_value, time.monotonic()-started, steps,
                     int(torch.count_nonzero(values != space.w0)), int(observed.sum()))
