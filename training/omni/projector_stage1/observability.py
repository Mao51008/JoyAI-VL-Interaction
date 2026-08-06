"""Dependency-free training observability helpers."""
from __future__ import annotations

import json
from pathlib import Path


def validation_is_due(
    completed_step: int,
    validate_every_steps: int,
    steps_per_epoch: int | None,
    has_validation: bool,
) -> bool:
    """Return whether a validation pass should run after a completed step."""
    if not has_validation:
        return False
    return completed_step % validate_every_steps == 0 or (
        steps_per_epoch is not None and completed_step % steps_per_epoch == 0
    )


def write_loss_curve(metrics_path: Path, output_path: Path) -> None:
    """Write a small self-contained SVG train/validation loss curve."""
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("metrics file contains no records")
    train = [(int(record["step"]), float(record["loss"])) for record in records]
    validation = [
        (int(record["step"]), float(record["validation_loss"]))
        for record in records
        if record.get("validation_loss") is not None
    ]
    values = [value for _, value in train + validation]
    min_value = min(values)
    max_value = max(values)
    value_span = max(max_value - min_value, 1e-6)
    max_step = max(step for step, _ in train)
    width, height = 800, 400
    left, right, top, bottom = 55, 20, 20, 45

    def point(step: int, value: float) -> str:
        x = left + (width - left - right) * step / max(max_step, 1)
        y = top + (height - top - bottom) * (max_value - value) / value_span
        return f"{x:.2f},{y:.2f}"

    train_points = " ".join(point(step, value) for step, value in train)
    validation_points = " ".join(point(step, value) for step, value in validation)
    validation_line = (
        f'<polyline fill="none" stroke="#d62728" stroke-width="2" points="{validation_points}"/>'
        if validation_points
        else ""
    )
    validation_legend = (
        '<text x="650" y="48" font-family="sans-serif" font-size="12" fill="#d62728">validation</text>'
        if validation_points
        else ""
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#444"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#444"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{train_points}"/>
{validation_line}
<text x="{left}" y="18" font-family="sans-serif" font-size="14">Projector loss</text>
<text x="{left}" y="{height-10}" font-family="sans-serif" font-size="12">step</text>
<text x="{width-150}" y="30" font-family="sans-serif" font-size="12" fill="#1f77b4">train</text>
{validation_legend}
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")
