"""Gate an isolated FHO100 derivative benchmark behind an exact FHO3 check.

This harness is intentionally solver-free: its child benchmark intercepts the
call to IPOPT immediately after callback construction.  It is intended for a
future hybrid callback which evaluates ``H_other`` natively and scatters the
exact compiled STATE_CONTINUITY contribution.  Today it can exercise the same
protocol with any candidate supported by
``benchmark_external_state_continuity_global.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


CALLBACKS = ("nlp_f", "nlp_g", "nlp_jac_g", "nlp_hess_l")


def fho3_gate(report: dict, *, tolerance: float, minimum_hessian_speedup: float) -> tuple[bool, list[str]]:
    """Return a deterministic go/no-go decision before allocating FHO100."""

    warm = report.get("warm", report)
    functions = warm.get("functions", {})
    reasons = []
    for name in CALLBACKS:
        row = functions.get(name)
        if row is None:
            reasons.append(f"missing {name}")
            continue
        if row.get("native_nnz") != row.get("external_nnz"):
            reasons.append(f"{name} sparsity differs")
        if row.get("max_abs_error", float("inf")) > tolerance:
            reasons.append(f"{name} error exceeds {tolerance:g}")
    hessian = functions.get("nlp_hess_l", {})
    if hessian.get("speedup", 0.0) < minimum_hessian_speedup:
        reasons.append(
            f"nlp_hess_l speedup {hessian.get('speedup')} is below {minimum_hessian_speedup:g}"
        )
    return not reasons, reasons


def _run_candidate(command_log: Path, output: Path, repeats: int, packet_rows: int | None) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_external_state_continuity_global.py")),
        "--command-log", str(command_log.resolve()),
        "--output", str(output.resolve()),
        "--repeats", str(repeats),
    ]
    if packet_rows is not None:
        command += ["--external-max-output-rows", str(packet_rows)]
    subprocess.run(command, check=True)
    return json.loads((output / "report.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fho3-command-log", type=Path, required=True)
    parser.add_argument("--fho100-command-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--packet-rows", type=int)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--minimum-hessian-speedup", type=float, default=1.05)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fho3 = _run_candidate(args.fho3_command_log, args.output / "fho3", args.repeats, args.packet_rows)
    accepted, reasons = fho3_gate(
        fho3, tolerance=args.tolerance, minimum_hessian_speedup=args.minimum_hessian_speedup
    )
    report = {
        "protocol": "FHO3 exact gate before isolated FHO100; IPOPT is never executed",
        "thresholds": {
            "max_abs_error": args.tolerance,
            "minimum_hessian_speedup": args.minimum_hessian_speedup,
        },
        "fho3": fho3,
        "fho3_accepted": accepted,
        "gate_reasons": reasons,
        "fho100_started": False,
    }
    if accepted:
        report["fho100"] = _run_candidate(args.fho100_command_log, args.output / "fho100", args.repeats, args.packet_rows)
        report["fho100_started"] = True
    (args.output / "protocol-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
