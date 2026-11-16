"""Recompute the sixteen scores in the two displayed visual cases."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    count=0
    for name in ["make3d","spacenet"]:
        metadata=json.loads((ROOT/"results/cases"/(name+".json")).read_text())
        with np.load(ROOT/"results/cases"/(name+".npz"),allow_pickle=False) as data:
            target=data["target"].astype(np.float64)
            mask=data["valid_mask"]
            for method,expected in metadata["scores"].items():
                pred=data["prediction__"+method].astype(np.float64)
                if metadata["metric"]=="AbsRel":
                    value=float(np.mean(np.abs(np.clip(pred[mask],.001,80)-target[mask])/target[mask]))
                else:
                    a,b=pred[mask]>0,target[mask]>0
                    value=float(2*np.count_nonzero(a & b)/max(np.count_nonzero(a)+np.count_nonzero(b),1))
                assert abs(value-expected)<=1e-6,(name,method,value,expected)
                count+=1
    print("Recomputed",count,"visual-case scores from prediction and target arrays.")


if __name__=="__main__":
    main()
