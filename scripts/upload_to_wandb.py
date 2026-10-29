"""Upload finished runs' JSONL logs to wandb, for runs trained without wandb.

Replays runs/<run_name>.jsonl into a new wandb run with the same name, config
and metric keys that MetricsLogger would have logged live, so the run shows up
next to the others in the project.

Usage:
    uv run python scripts/upload_to_wandb.py p2b_number_search_grpo p3_number_search_async \
        --project 11768-a3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb
import yaml

from scottyrl.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run names under runs/")
    parser.add_argument("--project", default="11768-a3", help="wandb project")
    args = parser.parse_args()

    required = yaml.safe_load((ROOT / "configs" / "required_runs.yaml").read_text())
    for run_name in args.runs:
        log_path = ROOT / "runs" / f"{run_name}.jsonl"
        config_path = ROOT / required.get(run_name, f"configs/{run_name}.yaml")
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        # An async run's background eval for step N lands in the log after
        # later training steps, and wandb drops any log whose step is below
        # the last one it saw. A stable sort puts it back at step N.
        records.sort(key=lambda record: record["step"])

        run = wandb.init(
            project=args.project,
            name=run_name,
            config=load_config(str(config_path), []).to_dict(),
            tags=["uploaded-from-jsonl"],
            notes=f"Replayed from runs/{run_name}.jsonl after training.",
        )
        # MetricsLogger logs every record except its "step" and "time" fields.
        for record in records:
            metrics = {k: v for k, v in record.items() if k not in ("step", "time")}
            run.log(metrics, step=record["step"])
        run.finish()
        print(f"uploaded {len(records)} records from {log_path.name} to {run.url}")


if __name__ == "__main__":
    main()
