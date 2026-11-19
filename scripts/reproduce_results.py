"""Reconstruct the paper tables and controlled comparisons from exact scores."""
from pathlib import Path
import argparse
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["Base","Adam","Memory","Calib","HypoLoop","HL-Auto","HL-LLM",
           "HL-Fixed","HL-Once","HL-Local","HL-Global"]


def reproduce(output):
    with (ROOT/"results/metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    tasks = json.loads((ROOT/"configs/tasks.json").read_text())
    lookup = {(r["task"],r["provider"],r["method"]):r for r in rows}
    assert len(rows)==len(lookup)==22*2*11
    changes = {}
    for task in tasks:
        for provider in ["deepseek","claude"]:
            block = {m:lookup[task["id"],provider,m] for m in METHODS}
            controls = {m:float(block[m]["score"]) for m in ["Adam","Memory","Calib"]}
            choose = max if block["Base"]["direction"]=="higher" else min
            reference = choose(controls.values())
            for method,row in block.items():
                score = float(row["score"])
                gain = 100*(score-reference) if row["direction"]=="higher" else 100*(1-score/reference)
                assert math.isclose(gain,float(row["improvement"]),abs_tol=1e-9,rel_tol=1e-10)
                assert math.isclose(reference,controls[row["feedback_control"]],abs_tol=1e-12)
                changes[task["id"],provider,method] = gain
    with (ROOT/"results/where.csv").open() as stream:
        where = list(csv.DictReader(stream))
    for r in where:
        assert math.isclose(100*(1-float(r["dynamic"])/float(r["fixed"])),
                            float(r["gain_percent"]),abs_tol=1e-9)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for table,field in [("table1","improvement"),("table2","score")]:
        lines=["| Task | Service | "+" | ".join(METHODS)+" |",
               "| :--- | :--- | "+" | ".join(["---:"]*len(METHODS))+" |"]
        for task in tasks:
            for provider in ["deepseek","claude"]:
                values=[]
                for method in METHODS:
                    row=lookup[task["id"],provider,method]
                    value=float(row[field])
                    values.append(f"{value:+.1f}" if field=="improvement" else
                                  f"{value:.{task['decimal_places']}f}")
                lines.append("| "+task["id"]+" | "+provider+" | "+" | ".join(values)+" |")
        (output/(table+".md")).write_text("\n".join(lines)+"\n")
    report=dict(tasks=22,services=2,methods=11,native_scores=484,matched_where_comparisons=len(where))
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="outputs/tables")
    reproduce(parser.parse_args().output)
