import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_hybrid_hessian_protocol.py"
_SPEC = importlib.util.spec_from_file_location("hybrid_hessian_protocol", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _report(error=0.0, speedup=1.2, nnz=10):
    return {
        "warm": {
            "functions": {
                name: {"max_abs_error": error, "speedup": speedup, "native_nnz": nnz, "external_nnz": nnz}
                for name in _MODULE.CALLBACKS
            }
        }
    }


def test_fho3_gate_accepts_exact_hessian_speedup():
    assert _MODULE.fho3_gate(_report(), tolerance=1e-10, minimum_hessian_speedup=1.05) == (True, [])


def test_fho3_gate_blocks_fho100_on_error_or_regression():
    accepted, reasons = _MODULE.fho3_gate(_report(error=1e-8, speedup=0.9), tolerance=1e-10, minimum_hessian_speedup=1.05)
    assert not accepted
    assert any("error" in item for item in reasons)
    assert any("speedup" in item for item in reasons)


def test_fho3_gate_blocks_non_finite_callback_audit():
    report = _report()
    report["warm"]["functions"]["nlp_g"]["max_abs_error"] = float("nan")
    accepted, reasons = _MODULE.fho3_gate(report, tolerance=1e-10, minimum_hessian_speedup=1.05)
    assert not accepted
    assert any("nlp_g error" in item for item in reasons)
