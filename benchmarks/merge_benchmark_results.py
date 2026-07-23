"""Merge isolated solver benchmark outputs into one artifact and Markdown table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CASE_ORDER = {
    "pendulum": 0,
    "cube": 1,
    "static_arm": 2,
    "free_time": 3,
    "multiphase": 4,
    "contact_inequality": 5,
    "holonomic_muscle": 6,
    "muscle_fatigue": 7,
}
SOLVER_ORDER = {"ipopt": 0, "fatrop": 1, "acados": 2, "madnlp": 3}


def optional_number(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def scientific(value: float | None) -> str:
    return "—" if value is None else f"{value:.2e}"


def load_results(input_directory: Path) -> tuple[list[dict], list[dict], list[dict]]:
    metadata: list[dict] = []
    summaries: list[dict] = []
    runs: list[dict] = []
    for path in sorted(input_directory.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not {"metadata", "summary", "runs"} <= payload.keys():
            continue
        if not isinstance(payload["metadata"], dict):
            continue
        metadata.append({"source": str(path), **payload["metadata"]})
        summaries.extend(payload["summary"])
        runs.extend(payload["runs"])
    if not summaries:
        raise RuntimeError(f"No benchmark JSON files found below {input_directory}")
    summaries.sort(
        key=lambda row: (
            CASE_ORDER.get(row["case"], len(CASE_ORDER)),
            row["n_shooting"],
            SOLVER_ORDER.get(row["solver"], len(SOLVER_ORDER)),
        )
    )
    return metadata, summaries, runs


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summaries: list[dict], metadata: list[dict]) -> None:
    platforms = sorted({row.get("platform", "unknown") for row in metadata})
    casadi_versions = sorted({row.get("casadi", "unknown") for row in metadata})
    lines = [
        "# Linux solver benchmark",
        "",
        f"- Platform: {', '.join(platforms)}",
        f"- CasADi: {', '.join(casadi_versions)}",
        "- Cold: first solve in a fresh Python process.",
        "- Hot: solve after one warm-up in the same process; this is not an OCP warm start.",
        "",
        "| Case | Shooting | Solver | Cold solve (s) | Hot solve (s) | Hot solver (s) | Hot iter. | Cost | Max violation | Outcome |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        successful = row["successful_runs"]
        outcome = "success" if successful else "failure"
        lines.append(
            "| "
            + " | ".join(
                (
                    row["case"],
                    str(row["n_shooting"]),
                    row["solver"],
                    optional_number(row.get("cold_solve_wall_s")),
                    optional_number(row.get("hot_solve_wall_s")),
                    optional_number(row.get("hot_solver_s")),
                    optional_number(row.get("hot_iterations"), digits=0),
                    optional_number(row.get("cost"), digits=6),
                    scientific(row.get("max_constraint_violation")),
                    outcome,
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata, summaries, runs = load_results(args.input_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata, "summary": summaries, "runs": runs}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output.with_suffix(".csv"), runs)
    write_markdown(args.output.with_suffix(".md"), summaries, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
