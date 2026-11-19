"""Recorded and numerical plans share the same executable editing interface."""
from dataclasses import replace
from .plans import EditPlan, FitSpec
from .programs import numeric_extra
import numpy as np


class RecordedPlanner:
    def __init__(self, plans):
        self.plans, self.index = list(plans), 0

    def propose(self, packet):
        plan = self.plans[min(self.index, len(self.plans)-1)]
        self.index += 1
        return plan


class FixedPlanner(RecordedPlanner):
    def __init__(self, plan):
        super().__init__([plan])


class NumericPlanner:
    def __init__(self, selectors, seed=17):
        self.selectors = tuple(selectors)
        self.rng = np.random.default_rng(seed)

    def propose(self, packet):
        records = [r for h in packet.get("history", []) for r in h.get("candidates", [])]+packet.get("candidate_results", [])
        previous = [r for r in records if r.get("cv_loss") is not None]
        specs = numeric_extra(previous, packet["binary"], {"full": ()}, ["all"], self.rng, count=1)
        return EditPlan(self.selectors, FitSpec(**specs[0]),
                        "Use the best current validation evidence to adapt the fitting rule.",
                        "Improve the next prediction batch while preserving the reference response.")
