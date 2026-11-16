from pathlib import Path
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from hypoloop import NativeModel, EditPlan, FitSpec, FeedbackEditor
from hypoloop.planners import RecordedPlanner
from hypoloop.linearized import fit_linearized
from hypoloop.metrics import grouped_folds
from hypoloop.transport import APIConfig, APIPlanner


class RevisionTests(unittest.TestCase):
    def test_stage_feedback_and_twelve_candidates(self):
        torch.set_num_threads(1)
        model=torch.nn.Linear(1,1).double()
        x=torch.arange(4,dtype=torch.float64).reshape(-1,1)/4
        native=NativeModel(model,lambda m,ids:m(x[ids]))
        target=native.predict([0,1,2,3]).numpy()+0.2
        class Planner:
            def __init__(self):
                self.seen=[]
            def propose(self,packet):
                self.seen.append(len(packet["candidate_results"]))
                return EditPlan(("bias",),FitSpec())
        planner=Planner()
        editor=FeedbackEditor(native,planner)
        event=editor.update([0,1,2,3],target,[(np.array([0,1]),np.array([2,3])),
                                             (np.array([2,3]),np.array([0,1]))])
        self.assertEqual(planner.seen,[0,6])
        self.assertEqual(len(event["candidates"]),12)
        self.assertIsNotNone(event["selected"])

    def test_unsupported_plan_retains_original(self):
        model=torch.nn.Linear(1,1).double()
        x=torch.ones(2,1,dtype=torch.float64)
        native=NativeModel(model,lambda m,ids:m(x[ids]))
        plan=EditPlan((),response_kind="unsupported")
        editor=FeedbackEditor(native,RecordedPlanner([plan]))
        target=native.predict([0,1]).numpy()+1
        event=editor.update([0,1],target,[(np.array([0]),np.array([1]))])
        self.assertIsNone(event["selected"])
        self.assertTrue(torch.equal(native.predict([0,1]),editor.predict([0,1])))

    def test_linearized_proposal_is_native_validated(self):
        model=torch.nn.Linear(1,1,bias=False).double()
        x=torch.ones(3,1,dtype=torch.float64)
        native=NativeModel(model,lambda m,ids:m(x[ids]))
        space=native.select(["weight"])
        y=space.predict([0,1,2],space.w0).numpy()+0.5
        result=fit_linearized(space,[0,1,2],y,[],FitSpec(ridge=0.001))
        self.assertLess(np.mean((space.predict([0,1,2],result.values).numpy()-y)**2),1e-4)
        native.check_original()

    def test_api_reply_recorded_without_authentication_material(self):
        value=EditPlan(("weight",),FitSpec()).to_dict()
        response=json.dumps(dict(model="returned-model",choices=[dict(message=dict(content=json.dumps(value)))])).encode()
        with tempfile.TemporaryDirectory() as folder:
            client=APIPlanner(APIConfig("deepseek","https://example.invalid","requested-model","private-key"),folder)
            with patch("urllib.request.urlopen",return_value=io.BytesIO(response)):
                plan=client.propose(dict(task_contract={"prediction_space":"regression"}))
            self.assertEqual(plan.tensor_names,("weight",))
            records=list(Path(folder).rglob("*"))
            self.assertTrue(any(p.name=="plan.json" for p in records))
            self.assertTrue(all("private-key" not in p.read_text() for p in records if p.is_file()))


if __name__=="__main__":
    unittest.main()
