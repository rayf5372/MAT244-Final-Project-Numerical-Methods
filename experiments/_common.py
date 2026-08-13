from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt 

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

OUTCOME_COLOURS = {
    "collision": "#c1272d",
    "bounded": "#0b6e4f",
    "escape": "#2b5d9e",
    "undetermined": "#8c8c8c",
}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.4,
        "figure.constrained_layout.use": True,
    }
)


def save_figure(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path.relative_to(ROOT)}")
    return path


def save_table(rows: list[dict], name: str) -> Path:
    """Write a list of uniform dicts as CSV."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  table  -> {path.relative_to(ROOT)}")
    return path


def save_text(text: str, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(text)
    print(f"  report -> {path.relative_to(ROOT)}")
    return path


class Report:

    def __init__(self, title: str):
        self.lines: list[str] = []
        rule = "=" * len(title)
        self(rule)
        self(title)
        self(rule)

    def __call__(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"