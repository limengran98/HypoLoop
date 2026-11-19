"""The native, fixed-plan, numerical, and one-update editing configurations."""
from .loop import FeedbackEditor
from .plans import EditPlan, FitSpec
from .planners import FixedPlanner, NumericPlanner
from .linearized import fit_linearized


METHODS = ("Base","Adam","Memory","Calib","HypoLoop","HL-Auto","HL-LLM",
           "HL-Fixed","HL-Once","HL-Local","HL-Global")


class InitialPlan:
    def __init__(self, planner):
        self.planner, self.plans = planner, {}

    def propose(self, packet):
        stage = packet["task_contract"]["stage"]
        if stage not in self.plans:
            self.plans[stage] = self.planner.propose(packet)
        return self.plans[stage]


class OnceEditor(FeedbackEditor):
    def update(self, *args, **kwargs):
        if self.history:
            return dict(updated=False, retained_round=0)
        return super().update(*args, **kwargs)


def make_editor(method, native, planner=None, *, local_selectors=(), **kwargs):
    if method not in METHODS[1:] or method in ("Memory","Calib"):
        raise ValueError("Use the output controls for Base, Memory and Calib.")
    local = tuple(local_selectors)
    if method != "HL-Global" and not local:
        raise ValueError("This configuration requires the API-selected local pool.")
    if method in ("Adam","HypoLoop","HL-Once"):
        planner = FixedPlanner(EditPlan(local, FitSpec(loss="bce" if native.binary else "mse")))
    elif method == "HL-Auto":
        planner = NumericPlanner(local)
    elif planner is None:
        raise ValueError("Supply an API or recorded planner.")
    if method == "HL-Fixed":
        planner = InitialPlan(planner)
    if method not in ("Adam","HL-Global","HL-Local"):
        kwargs["fitter"] = fit_linearized
    editor = OnceEditor if method == "HL-Once" else FeedbackEditor
    return editor(native, planner, fixed_selectors=None if method == "HL-Global" else local, **kwargs)
