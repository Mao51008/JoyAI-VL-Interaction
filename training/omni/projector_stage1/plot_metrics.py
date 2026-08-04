"""Plot projector stage-one metrics.jsonl and write the best validation summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> list[dict]:
    """Load non-empty JSONL records and reject malformed training metrics."""
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
        if not isinstance(record, dict) or "step" not in record or "loss" not in record or "learning_rate" not in record:
            raise ValueError(f"metrics record on line {line_number} is missing required fields")
        records.append(record)
    if not records:
        raise ValueError("metrics file contains no records")
    return records


def best_validation_summary(records: list[dict]) -> dict:
    validation_records = [record for record in records if "validation_loss" in record]
    if not validation_records:
        raise ValueError("metrics file contains no validation_loss records")
    best = min(validation_records, key=lambda record: float(record["validation_loss"]))
    return {"best_validation_step": best["step"], "best_validation_loss": best["validation_loss"],
            "learning_rate": best["learning_rate"], "training_loss": best["loss"]}


def plot_metrics(records: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    steps = [record["step"] for record in records]
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    axes[0].plot(steps, [record["loss"] for record in records], label="train loss")
    validation = [record for record in records if "validation_loss" in record]
    axes[1].plot([record["step"] for record in validation], [record["validation_loss"] for record in validation], marker="o", label="validation loss")
    axes[2].plot(steps, [record["learning_rate"] for record in records], label="learning rate")
    for axis in axes:
        axis.legend(); axis.grid(True, alpha=0.3)
    axes[2].set_xlabel("step")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    records = load_metrics(args.metrics)
    summary = best_validation_summary(records)
    plot_metrics(records, args.output_png)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
