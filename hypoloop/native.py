"""Native coordinates, functional inference, and portable weight patches."""
from pathlib import Path
import hashlib
import re
import torch
from torch.func import functional_call


def state_hash(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


class _Forward(torch.nn.Module):
    def __init__(self, model, predict):
        super().__init__()
        self.model, self.predict = model, predict

    def forward(self, ids):
        return self.predict(self.model, ids)


class NativeModel:
    """A complete predictor with a user-supplied native serving function.

    predict(model, ids) returns an object-by-output tensor. It retains the
    model's original preprocessing, nonlinearities, and output transforms.
    """
    def __init__(self, model, predict, *, output_scale=1.0, binary=False):
        self.model = model.eval()
        self.predict_module = predict
        self.output_scale, self.binary = float(output_scale), bool(binary)
        if not 0 < self.output_scale < float("inf"):
            raise ValueError("output_scale must be positive.")
        self.wrapper = _Forward(self.model, predict)
        self.parameters = dict(model.named_parameters(remove_duplicate=False))
        self.canonical, self.aliases = {}, {}
        physical = {}
        for name, p in self.parameters.items():
            physical.setdefault(id(p), name)
            canonical = physical[id(p)]
            self.canonical[name] = canonical
            self.aliases.setdefault(canonical, []).append(name)
        self.original_state = state_hash(model)
        self.device = next(model.parameters()).device

    def inventory(self):
        return [dict(name=name, shape=list(p.shape), numel=p.numel(),
                     real_coordinates=p.numel() * (2 if p.is_complex() else 1),
                     dtype=str(p.dtype), alias_of=self.canonical[name]
                     if self.canonical[name] != name else None)
                for name, p in self.parameters.items()
                if p.is_floating_point() or p.is_complex()]

    def select(self, selectors, budget=8192):
        return ParameterSpace(self, selectors, budget)

    def predict(self, ids):
        with torch.no_grad():
            return self.predict_module(self.model, list(ids)).detach().double()

    def check_original(self):
        if state_hash(self.model) != self.original_state:
            raise RuntimeError("Native inference changed the original state.")


class ParameterSpace:
    def __init__(self, native, selectors, budget):
        self.native, self.selectors = native, tuple(selectors)
        self.device, self.blocks = native.device, []
        values, scales, position = [], [], 0
        for selector in selectors:
            m = re.fullmatch(r"(.+)::flat\[(\d+):(\d+)\]", selector)
            requested = m[1] if m else selector
            if requested not in native.parameters:
                raise ValueError("Unknown parameter: " + requested)
            name = native.canonical[requested]
            p = native.parameters[name]
            if not (p.is_floating_point() or p.is_complex()):
                raise ValueError("Only real and complex parameters are editable.")
            flat = self.flatten(p.detach())
            start, stop = (int(m[2]), int(m[3])) if m else (0, flat.numel())
            if not 0 <= start < stop <= flat.numel():
                raise ValueError("Coordinate block is outside the parameter.")
            for b in self.blocks:
                if b["name"] == name and max(start, b["start"]) < min(stop, b["stop"]):
                    raise ValueError("Overlapping coordinates or tied aliases.")
            value = flat[start:stop].double().clone()
            scale = max(0.01, float(value.square().mean().sqrt()))
            self.blocks.append(dict(name=name, start=start, stop=stop,
                                    begin=position, end=position+stop-start))
            values.append(value)
            scales.append(torch.full_like(value, scale))
            position += stop-start
        if not 0 < position <= budget or len(selectors) > 8:
            raise ValueError("An edit requires at most eight blocks within its budget.")
        self.dimension = position
        self.w0, self.scales = torch.cat(values), torch.cat(scales)

    @staticmethod
    def flatten(p):
        return torch.view_as_real(p).reshape(-1) if p.is_complex() else p.reshape(-1)

    def replacements(self, values):
        if values.shape != self.w0.shape:
            raise ValueError("The patch dimension differs from its selected space.")
        result = {}
        for name in dict.fromkeys(b["name"] for b in self.blocks):
            p = self.native.parameters[name]
            flat = self.flatten(p.detach())
            for b in self.blocks:
                if b["name"] == name:
                    ix = torch.arange(b["start"], b["stop"], device=self.device)
                    flat = flat.scatter(0, ix, values[b["begin"]:b["end"]].to(flat.dtype))
            tensor = torch.view_as_complex(flat.reshape(*p.shape, 2)) if p.is_complex() else flat.reshape(p.shape)
            for alias in self.native.aliases[name]:
                result[alias] = tensor
        return result

    def forward(self, ids, values):
        replacement = {"model."+k: v for k, v in self.replacements(values).items()}
        result = functional_call(self.native.wrapper, replacement, (list(ids),),
                                 strict=False, tie_weights=False)
        if result.ndim != 2 or len(result) != len(ids):
            raise ValueError("The serving function must return an object-by-output tensor.")
        return result

    def predict(self, ids, values):
        with torch.no_grad():
            return self.forward(ids, values).detach().double()

    def pack(self, values):
        replaced = self.replacements(values)
        return torch.cat([self.flatten(replaced[b["name"]].detach())[b["start"]:b["stop"]].double()
                          for b in self.blocks])

    def save(self, path, values):
        self.native.check_original()
        state = {n: v.detach().cpu().clone() for n, v in self.native.model.state_dict().items()}
        for name, value in self.replacements(self.pack(values)).items():
            state[name] = value.detach().cpu()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model_state_dict=state, selectors=self.selectors,
                        selected_coordinates=self.dimension,
                        original_state=self.native.original_state), path)
        return path
