"""Feedback-driven relocation, objective revision, validation, and deployment."""
from dataclasses import asdict
from pathlib import Path
import json
import inspect
import hashlib
import numpy as np
from .metrics import mse
from .plans import EditPlan, FitSpec
from .programs import anchors, fixed_extra
from .solver import fit_patch


class FeedbackEditor:
    """The two-stage native editing loop described in the method section."""
    def __init__(self, native, planner, *, reference_ids=(), metric=mse,
                 parameter_budget=8192, minimum_cv_gain=0.001,
                 maximum_history_degradation=0.05, validate=None,
                 fixed_selectors=None, output=None, fitter=fit_patch):
        self.native, self.planner = native, planner
        self.reference_ids, self.metric = tuple(reference_ids), metric
        self.parameter_budget = parameter_budget
        self.minimum_cv_gain = minimum_cv_gain
        self.maximum_history_degradation = maximum_history_degradation
        self.validate = validate
        self.fixed_selectors = fixed_selectors
        self.fit = fitter
        self.output = Path(output) if output else None
        self.current, self.history = None, []

    def predict(self, ids):
        if self.current is None:
            return self.native.predict(ids)
        space, values = self.current
        return space.predict(ids, values)

    def _evaluate(self, selectors, spec, ids, target, folds, number):
        space = self.native.select(selectors, self.parameter_budget)
        predictions, targets, names = [], [], []
        costs = []
        for train, valid in folds:
            train, valid = np.asarray(train), np.asarray(valid)
            if not len(train) or not len(valid) or np.intersect1d(train, valid).size:
                raise ValueError("Fitting and validation identities must be disjoint.")
            fit_ids, valid_ids = [ids[i] for i in train], [ids[i] for i in valid]
            result = self.fit(space, fit_ids, target[train], self.reference_ids, spec,
                              ages=(train.max()-train)/max(float(train.max()-train.min()),1.0))
            predictions.append(space.predict(valid_ids, result.values).cpu().numpy())
            targets.append(target[valid])
            names.extend(valid_ids)
            costs.append(result.seconds)
        cv = self.metric(np.concatenate(predictions), np.concatenate(targets), names)
        result = self.fit(space, ids, target, self.reference_ids, spec)
        pred = space.predict(ids, result.values).cpu().numpy()
        feasible = True
        if self.validate:
            check_ids = list(ids)+list(self.reference_ids)
            feasible = bool(self.validate(space.predict(check_ids, result.values).cpu().numpy(), check_ids))
        report = dict(candidate=number, selectors=list(selectors), spec=asdict(spec),
                      cv_loss=cv, feedback_loss=self.metric(pred, target, ids),
                      changed_coordinates=result.changed_coordinates,
                      selected_coordinates=space.dimension, seconds=sum(costs)+result.seconds,
                      feasible=feasible, status="complete")
        return report, space, result.values

    def update(self, ids, target, folds, *, stages=2):
        ids, target = list(ids), np.asarray(target)
        if len(ids) != len(target) or len(set(ids)) != len(ids):
            raise ValueError("Feedback identities must be unique and aligned with targets.")
        folds = list(folds)
        if not folds:
            raise ValueError("Use explicit grouped or temporal feedback folds.")
        folder = None
        if self.output:
            folder = self.output/f"round_{len(self.history)+1:02d}"
            folder.mkdir(parents=True, exist_ok=False)
            (folder/"started.json").write_text(json.dumps(dict(
                ids=ids, target_sha256=hashlib.sha256(target.tobytes()).hexdigest())))
        pairs = [int(i) for _, valid in folds for i in valid]
        baseline = self.native.predict([ids[i] for i in pairs]).cpu().numpy()
        cv_base = self.metric(baseline, target[pairs], [ids[i] for i in pairs])
        current_loss = self.metric(self.predict(ids).cpu().numpy(), target, ids)
        complete, reports, plans = [], [], []
        p0 = self.native.predict(ids).cpu().numpy()
        observed = np.isfinite(target)
        residual = (p0-target)[observed]
        try:
            source = inspect.getsource(type(self.native.model))[:48000]
        except (OSError, TypeError):
            source = type(self.native.model).__module__+"."+type(self.native.model).__qualname__
        for stage in range(stages):
            packet = dict(model=dict(parameters=self.native.inventory(), source=source),
                          binary=self.native.binary,
                          task_contract=dict(parameter_budget=self.parameter_budget,
                                             prediction_space="binary_logits" if self.native.binary else "regression",
                                             output_scale=self.native.output_scale,
                                             conditions=["all"], stage=stage, round=len(self.history)),
                          known_feedback=dict(objects=len(ids), observed_outputs=int(observed.sum()),
                                              signed_error=float(residual.mean()),
                                              mae=float(np.abs(residual).mean()),
                                              rmse=float(np.sqrt((residual**2).mean()))),
                          candidate_results=reports, history=self.history)
            plan = self.planner.propose(packet)
            if isinstance(plan, dict):
                plan = EditPlan.from_dict(plan)
            plans.append(plan.to_dict())
            if plan.response_kind == "unsupported":
                continue
            selectors = self.fixed_selectors or plan.tensor_names
            space = self.native.select(selectors, self.parameter_budget)
            selected = {b["name"] for b in space.blocks}
            protected = {self.native.canonical[n] for n in plan.protected_tensors}
            if selected & protected:
                raise ValueError("Selected coordinates overlap protected parameters.")
            specs = [FitSpec(**s) for s in anchors(self.native.binary)]
            specs += [plan.fit_spec, FitSpec(**fixed_extra(self.native.binary)[0])]
            for spec in specs:
                try:
                    record, space, values = self._evaluate(selectors, spec, ids, target, folds, len(reports))
                    complete.append((record, space, values))
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    record = dict(candidate=len(reports), selectors=list(selectors),
                                  spec=asdict(spec), status="failed", error=str(error))
                reports.append(record)
        accepted = [item for item in complete
                    if item[0]["feasible"]
                    and item[0]["cv_loss"] < (1-self.minimum_cv_gain)*cv_base
                    and item[0]["feedback_loss"] <= (1+self.maximum_history_degradation)*current_loss]
        selected = min(accepted, key=lambda x: x[0]["cv_loss"]) if accepted else None
        if selected:
            record, space, values = selected
            self.current = space, values
        event = dict(round=len(self.history), feedback_objects=len(ids), plans=plans,
                     candidates=reports, selected=selected[0] if selected else None,
                     baseline_cv_loss=cv_base, previous_feedback_loss=current_loss)
        self.history.append(event)
        if folder:
            (folder/"selection.json").write_text(json.dumps(event, indent=2))
            if self.current:
                self.current[0].save(folder/"model.pt", self.current[1])
        return event
