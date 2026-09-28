"""Compare native and opt-in External-C STATE_CONTINUITY on a complete MX NLP.

The runner command is intercepted immediately before IPOPT is called.  It
therefore builds precisely the production FHO graph but does not alter or run
any campaign.  Both NLPs are evaluated at the same initial point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shlex
import statistics
import sys
import time

import casadi as ca
import numpy as np


class _Captured(BaseException):
    pass


class _StopSolver:
    def __init__(self, solver):
        self.solver = solver

    def call(self, _):
        raise _Captured()

    def stats(self):
        return self.solver.stats()


def _command(path: Path) -> list[str]:
    return list(json.loads(path.read_text())) if path.suffix == ".json" else shlex.split(path.read_text())


def _replace(cli: list[str], flag: str, value: str):
    if flag in cli:
        cli[cli.index(flag) + 1] = value
    else:
        cli.extend((flag, value))


def _capture(command: list[str], output: Path, external_cache: Path | None):
    script, cli = Path(command[1]).resolve(), command[2:].copy()
    _replace(cli, "--output-json", str(output / "unused-result.json"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # The historical cycling launcher belongs to the parent cocofest project,
    # whereas this benchmark intentionally lives in the isolated Bioptim tree.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import bioptim.interfaces.interface_utils as iu
    import bioptim.interfaces.ipopt_interface as ii

    original_nlpsol, original_init = iu.nlpsol, ii.IpoptInterface.__init__
    captured = {}

    def patched_init(self, ocp):
        original_init(self, ocp)
        if external_cache is not None:
            self.enable_compiled_thread_map_external("STATE_CONTINUITY", external_cache)

    def patched_nlpsol(name, plugin, nlp, options):
        solver = original_nlpsol(name, plugin, nlp, options)
        captured["solver"] = solver
        return _StopSolver(solver)

    ii.IpoptInterface.__init__, iu.nlpsol = patched_init, patched_nlpsol
    sys.argv = [str(script), *cli]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except _Captured:
        pass
    finally:
        ii.IpoptInterface.__init__, iu.nlpsol = original_init, original_nlpsol
    return captured["solver"]


def _inputs(function, x, lam_g):
    values = []
    for index in range(function.n_in()):
        name = function.name_in(index)
        shape = function.size_in(index)
        if name in ("x", "i0") and shape[0] == x.numel():
            values.append(x)
        elif "lam_g" in name or (shape[0] == lam_g.numel() and shape[0] != x.numel()):
            values.append(lam_g)
        else:
            # sigma and unused multiplier inputs are zero for a structural
            # comparison.  State-continuity is in g, so this remains exact.
            values.append(ca.DM.zeros(*shape))
    return values


def _measure(function, values, repeats):
    samples, result = [], None
    for _ in range(repeats):
        tic = time.perf_counter()
        result = function(*values)
        samples.append(time.perf_counter() - tic)
    return statistics.median(samples), result


def _max_error(left, right):
    left = left if isinstance(left, (tuple, list)) else (left,)
    right = right if isinstance(right, (tuple, list)) else (right,)
    # ``nlp_hess_l`` is a 12k x 12k sparse matrix.  Converting it to a dense
    # NumPy array would allocate gigabytes just to audit equality.  Callback
    # sparsities are reported separately, hence comparing CasADi's stored
    # nonzero stream is both sufficient and memory bounded.
    def error(a, b):
        a_values, b_values = np.asarray(a.nonzeros()), np.asarray(b.nonzeros())
        if a_values.shape != b_values.shape:
            return float("inf")
        return float(np.max(np.abs(a_values - b_values))) if a_values.size else 0.0

    return max(error(a, b) for a, b in zip(left, right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = _command(args.command_log)
    native = _capture(command, args.output / "native", None)
    external = _capture(command, args.output / "external", args.output / "external-cache")
    x = ca.DM.zeros(native.size_in(0), 1)
    # IPOPT callback dimensions are solver-independent; use a nonzero g
    # multiplier vector to exercise the exact second derivative path.
    g = native.get_function("nlp_g")
    lam_g = ca.DM.ones(g.size_out(0), 1)
    report = {}
    for name in ("nlp_f", "nlp_g", "nlp_jac_g", "nlp_hess_l"):
        native_f, external_f = native.get_function(name), external.get_function(name)
        native_values = _inputs(native_f, x, lam_g)
        external_values = _inputs(external_f, x, lam_g)
        native_s, native_out = _measure(native_f, native_values, args.repeats)
        external_s, external_out = _measure(external_f, external_values, args.repeats)
        report[name] = {
            "native_median_s": native_s,
            "external_median_s": external_s,
            "speedup": native_s / external_s if external_s else None,
            "max_abs_error": _max_error(native_out, external_out),
            "native_nnz": native_f.sparsity_out(0).nnz(),
            "external_nnz": external_f.sparsity_out(0).nnz(),
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
