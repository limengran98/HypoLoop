"""The executable Where and How decisions."""
from dataclasses import asdict, dataclass, field
from .programs import validate_spec


@dataclass(frozen=True)
class FitSpec:
    loss: str = "mse"
    ridge: float = 0.01
    preservation: float = 0.0
    step: float = 1.0
    scope: str = "full"
    condition: str = "all"
    weight_program: str = "1"

    def validate(self, binary=False):
        validate_spec(asdict(self), {"full": ()}, ["all"], binary)
        return self


@dataclass(frozen=True)
class EditPlan:
    tensor_names: tuple[str, ...]
    fit_spec: FitSpec = field(default_factory=FitSpec)
    justification: str = ""
    expected_behavior: str = ""
    protected_tensors: tuple[str, ...] = ()
    response_kind: str = "local_linearization"

    @classmethod
    def from_dict(cls, value):
        required = {"tensor_names", "fit_spec", "justification", "expected_behavior",
                    "protected_tensors", "response_kind"}
        if set(value) != required:
            raise ValueError("An editing plan requires the six declared fields.")
        names=value["tensor_names"]
        if not isinstance(names,list) or len(names)>8 or len(names)!=len(set(names)) or any(not isinstance(n,str) for n in names):
            raise ValueError("Select at most eight unique native tensor names.")
        if value["response_kind"] not in ("exact_affine","local_linearization","unsupported"):
            raise ValueError("Unknown response kind.")
        if (value["response_kind"]=="unsupported") != (len(names)==0):
            raise ValueError("Unsupported plans have no editable tensors.")
        return cls(tuple(value["tensor_names"]), FitSpec(**value["fit_spec"]),
                   value["justification"], value["expected_behavior"],
                   tuple(value["protected_tensors"]), value["response_kind"])

    def to_dict(self):
        return asdict(self)
