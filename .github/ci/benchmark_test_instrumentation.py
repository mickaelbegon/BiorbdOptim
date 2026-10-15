"""Compare baseline and candidate in fresh pytest processes on the same runner.

Only the instrumented test module is swapped; source code, dependencies, coverage,
node IDs and assertions stay identical. The original candidate is always restored.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path("tests/shard2/test_global_getting_started.py")


def signature(path):
    cases = ET.parse(path).getroot().iter("testcase")
    return Counter(
        (
            case.get("classname"),
            case.get("name"),
            (
                "failed"
                if case.find("failure") is not None
                else (
                    "error"
                    if case.find("error") is not None
                    else "skipped" if case.find("skipped") is not None else "passed"
                )
            ),
        )
        for case in cases
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--order", default="baseline,candidate")
    parser.add_argument("--selection", default="")
    args = parser.parse_args()
    order = args.order.split(",")
    if set(order) != {"baseline", "candidate"}:
        parser.error("order must include baseline and candidate")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = ROOT / TARGET
    candidate = target.read_bytes()
    baseline = subprocess.check_output(["git", "show", f"{args.base}:{TARGET.as_posix()}"], cwd=ROOT)
    versions = subprocess.check_output(["conda", "list", "--prefix", sys.prefix, "--json"], text=True)
    (output / "packages.json").write_text(versions)
    report = {
        "base": args.base,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "platform": platform.platform(),
        "python": sys.version,
        "coverage": args.coverage,
        "selection": args.selection,
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GITHUB_ACTIONS")
        },
        "hashes": {
            name: hashlib.sha256(data).hexdigest() for name, data in [("baseline", baseline), ("candidate", candidate)]
        },
        "runs": [],
    }
    try:
        for index, name in enumerate(order, 1):
            target.write_bytes(baseline if name == "baseline" else candidate)
            # Source rewrites can keep the same size and timestamp granularity.
            for cached in target.parent.glob("__pycache__/test_global_getting_started.*.pyc"):
                cached.unlink()
            label = f"{index}-{name}"
            xml = output / f"{label}.xml"
            audit = output / f"{label}-tracing.json"
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--color=no",
                str(TARGET),
                "--durations=20",
                f"--junitxml={xml}",
                "-p",
                "tracing_probe",
            ]
            if args.coverage:
                command += [
                    "--cov=bioptim",
                    "--cov-report=term",
                    f"--cov-report=xml:{output / (label + '-coverage.xml')}",
                ]
            if args.selection:
                command += ["-k", args.selection]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
            env["TRACING_AUDIT_FILE"] = str(audit)
            print(f"Starting {label}", flush=True)
            start = time.perf_counter()
            with (output / f"{label}.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            row = {
                "label": label,
                "variant": name,
                "seconds": time.perf_counter() - start,
                "exit_code": result.returncode,
                "command": command,
                "tracing_leaks": json.loads(audit.read_text()) if audit.exists() else None,
            }
            report["runs"].append(row)
            (output / "results.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in row.items() if k not in ("command", "tracing_leaks")}), flush=True)
            print("\n".join((output / f"{label}.log").read_text().splitlines()[-24:]), flush=True)
    finally:
        target.write_bytes(candidate)
    signatures = [signature(output / f"{row['label']}.xml") for row in report["runs"]]
    report["identical_cases_and_outcomes"] = all(sig == signatures[0] for sig in signatures)
    report["case_count"] = sum(signatures[0].values())
    report["outcomes"] = dict(
        Counter(
            {
                status: sum(n for (_, _, state), n in signatures[0].items() if state == status)
                for status in ("passed", "skipped", "failed", "error")
            }
        )
    )
    report["valid"] = (
        report["identical_cases_and_outcomes"]
        and all(row["exit_code"] == 0 for row in report["runs"])
        and all(row["tracing_leaks"] == [] for row in report["runs"] if row["variant"] == "candidate")
    )
    (output / "results.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({key: value for key, value in report.items() if key not in ("runs", "hashes", "python")}), flush=True
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
