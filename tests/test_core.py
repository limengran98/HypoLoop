import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from hypoloop import NativeModel, EditPlan, FitSpec, FeedbackEditor, fit_patch
from hypoloop.metrics import average_precision, grouped_folds, mse
from hypoloop.planners import RecordedPlanner
from hypoloop.programs import run_program
from hypoloop.solver import normalized_weights
from hypoloop.stream import FeedbackStream
from hypoloop.transport import APIConfig, APIPlanner


class NativeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = torch.nn.Sequential(torch.nn.Linear(2,3), torch.nn.Tanh(),
                                         torch.nn.Linear(3,1), torch.nn.Sigmoid()).double()
        self.x = torch.tensor([[0.1,0.2],[0.5,-0.4],[-0.3,0.7]], dtype=torch.float64)
        self.native = NativeModel(self.model, lambda m, ids: m(self.x[ids]))

    def test_native_nonlinearity_retained(self):
        space = self.native.select(["2.bias"])
        values = space.w0+0.8
        actual = space.predict([0,1,2], values)
        clone = torch.nn.Sequential(torch.nn.Linear(2,3), torch.nn.Tanh(),
                                    torch.nn.Linear(3,1), torch.nn.Sigmoid()).double()
        clone.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            clone[2].bias.add_(0.8)
        self.assertTrue(torch.equal(actual, clone(self.x)))
        self.native.check_original()

    def test_outside_coordinates_restored(self):
        space = self.native.select(["0.weight::flat[0:2]"])
        values = space.w0+1
        changed = space.replacements(values)["0.weight"]
        self.assertTrue(torch.equal(changed.flatten()[2:], self.model[0].weight.flatten()[2:]))

    def test_budget_and_overlap(self):
        with self.assertRaises(ValueError):
            self.native.select(["0.weight"], budget=1)
        with self.assertRaises(ValueError):
            self.native.select(["0.weight::flat[0:3]","0.weight::flat[2:4]"])

    def test_checkpoint_replay(self):
        space = self.native.select(["2.weight","2.bias"])
        values = space.w0+0.2
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.pt"
            space.save(path, values)
            model = torch.nn.Sequential(torch.nn.Linear(2,3), torch.nn.Tanh(),
                                        torch.nn.Linear(3,1), torch.nn.Sigmoid()).double()
            model.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
            self.assertTrue(torch.equal(model(self.x), space.predict([0,1,2], values)))

    def test_true_nonlinear_fit_improves_training(self):
        space = self.native.select(["2.bias"])
        target = space.predict([0,1,2], space.w0).numpy()+0.1
        result = fit_patch(space, [0,1,2], target, [], FitSpec(ridge=0.0001))
        self.assertLess(mse(space.predict([0,1,2],result.values).numpy(),target), 0.001)
        self.assertEqual(result.steps,80)

    def test_complex_coordinates(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.coefficient = torch.nn.Parameter(torch.tensor([1+2j,3+4j],dtype=torch.complex128))
            def forward(self,x):
                return (x*self.coefficient).real.sum().reshape(1,1)
        model = Model()
        native = NativeModel(model,lambda m,ids: m(torch.ones(2,dtype=torch.complex128)))
        space = native.select(["coefficient::flat[1:3]"])
        self.assertEqual(space.dimension,2)
        self.assertTrue(torch.equal(space.w0,torch.tensor([2.,3.],dtype=torch.float64)))

    def test_tied_aliases(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.left=torch.nn.Linear(1,1,bias=False).double()
                self.right=torch.nn.Linear(1,1,bias=False).double()
                self.right.weight=self.left.weight
            def forward(self,x):
                return self.left(x)+self.right(x)
        model=Model()
        native=NativeModel(model,lambda m,ids:m(torch.ones(len(ids),1,dtype=torch.float64)))
        with self.assertRaises(ValueError):
            native.select(["left.weight","right.weight"])
        space=native.select(["left.weight"])
        replacements=space.replacements(space.w0+1)
        self.assertTrue(torch.equal(replacements["left.weight"],replacements["right.weight"]))


class EvidenceTests(unittest.TestCase):
    def test_restricted_weight_language(self):
        with self.assertRaises(ValueError):
            run_program("__import__('os').system('true')",{},(1,1))
        result=run_program("clip(exp(2*residual),0.2,5)",
                           dict(residual=np.array([[-1.,0.,1.]])),(1,3))
        np.testing.assert_allclose(result,[[0.2,1.,5.]])

    def test_masked_observations_have_zero_weight(self):
        y=np.array([[np.nan,1.],[2.,3.]])
        target,q,scale,observed=normalized_weights(np.ones((2,2)),y,1.,FitSpec())
        self.assertEqual(q[0,0],0)
        self.assertAlmostEqual(q.sum(),1)
        self.assertTrue(np.isfinite(target).all())

    def test_grouped_cv(self):
        ids=list("abcdef")
        groups=dict(zip(ids,["g1","g1","g2","g2","g3","g3"]))
        for train,valid in grouped_folds(ids,groups):
            self.assertFalse({groups[ids[i]] for i in train}&{groups[ids[i]] for i in valid})

    def test_average_precision_ties(self):
        self.assertAlmostEqual(average_precision([1,1,1,1],[0,1,0,1]),0.5)
        self.assertEqual(average_precision([1,0],[1,0]),1)

    def test_seal_precedes_target_access(self):
        calls=[]
        with tempfile.TemporaryDirectory() as folder:
            stream=FeedbackStream(["F0"],lambda role,ids:calls.append(role) or np.ones((1,1)),folder)
            with self.assertRaises(FileNotFoundError):
                stream.reveal("F0")
            self.assertFalse(calls)
            stream.seal("F0",["a"],{"Base":np.zeros((1,1))})
            stream.reveal("F0")
            self.assertEqual(calls,["F0"])

    def test_prediction_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            stream=FeedbackStream(["F0"],lambda role,ids:np.ones((1,1)),folder)
            stream.seal("F0",["a"],{"Base":np.zeros((1,1))})
            with (Path(folder)/"F0.npz").open("ab") as out:
                out.write(b"changed")
            with self.assertRaises(ValueError):
                stream.reveal("F0")

    def test_provider_message_formats(self):
        for style in ["openai","anthropic"]:
            with tempfile.TemporaryDirectory() as folder:
                client=APIPlanner(APIConfig("claude","https://example.invalid","test","private-key",style),folder)
                url,headers,body=client._request({"known_feedback":{"objects":3}})
                self.assertEqual(body["model"],"test")
                self.assertIn("messages",body)
                if style=="anthropic":
                    self.assertTrue(url.endswith("/v1/messages"))
                else:
                    self.assertTrue(url.endswith("/chat/completions"))


if __name__=="__main__":
    unittest.main()
