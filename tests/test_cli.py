import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from hypoloop import EditPlan, FitSpec
from hypoloop import cli
from hypoloop.tasks import load_bundle


class BundleTests(unittest.TestCase):
    def test_both_services_seal_before_query_labels(self):
        torch.set_num_threads(1)
        packets=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            bundle=root/"task"
            bundle.mkdir()
            model=torch.nn.Linear(1,1)
            with torch.no_grad():
                model.weight.zero_()
                model.bias.zero_()
            torch.save(model.state_dict(),bundle/"model.pt")
            ids=["a","b","c","d","q","r"]
            torch.save({g:torch.ones(1,1) for g in ids},bundle/"inputs.pt")
            np.savez(bundle/"F0.npz",ids=np.asarray(ids[:4]),y=np.ones((4,1)))
            np.savez(bundle/"E.npz",ids=np.asarray(["q"]),y=np.array([[888888.]]))
            manifest=dict(model_factory="torch.nn:Linear",model_kwargs=dict(in_features=1,out_features=1),
                          checkpoint="model.pt",inputs="inputs.pt",metric="mse",output_scale=1.,
                          reference_ids=["r"],roles={"F0":ids[:4],"E":["q"]},
                          targets={"F0":"F0.npz","E":"E.npz"},
                          folds={"4":[dict(train=[0,1],validation=[2,3]),dict(train=[2,3],validation=[0,1])]})
            (bundle/"manifest.json").write_text(json.dumps(manifest))
            output=root/"run"
            class Planner:
                def __init__(self,*args):
                    pass
                def propose(self,packet):
                    packets.append(json.loads(json.dumps(packet)))
                    return EditPlan(("bias",),FitSpec())
            def guarded_loader(*args):
                task=load_bundle(*args)
                original=task.targets
                def targets(role,role_ids):
                    record=json.loads((output/"predictions"/(role+".seal.json")).read_text())
                    self.assertEqual(record["predictors"],["claude","deepseek"])
                    return original(role,role_ids)
                task.targets=targets
                return task
            with patch("hypoloop.cli.APIConfig.from_env",return_value=None), \
                 patch("hypoloop.cli.APIPlanner",Planner), \
                 patch("hypoloop.cli.load_bundle",side_effect=guarded_loader):
                cli.run(argparse.Namespace(output=str(output),provider="both",bundle=str(bundle),device="cpu"))
            self.assertEqual(len(packets),4)
            self.assertTrue(all(p["known_feedback"]["objects"]==4 for p in packets))
            self.assertTrue(all("888888" not in json.dumps(p) for p in packets))
            for provider in ["deepseek","claude"]:
                self.assertTrue((output/provider/"checkpoints/E.pt").is_file())


if __name__=="__main__":
    unittest.main()
