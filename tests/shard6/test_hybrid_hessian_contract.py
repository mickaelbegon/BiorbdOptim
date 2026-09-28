import importlib.util
from pathlib import Path

import casadi as ca
import numpy as np


_SCRIPT = Path(__file__).parents[2] / "scripts" / "hybrid_hessian_contract.py"
_SPEC = importlib.util.spec_from_file_location("hybrid_hessian_contract", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _hessian(name, offset=0.0):
    x, p, sigma, lam = ca.MX.sym("x", 2), ca.MX.sym("p", 0, 0), ca.MX.sym("sigma"), ca.MX.sym("lam", 3)
    hessian = ca.diag(ca.vertcat(sigma + lam[0], 2 * lam[1],)) + offset
    return ca.Function(name, [x, p, sigma, lam], [hessian], ["x", "p", "sigma", "lam_g"], ["hess_gamma_x_x"])


def test_hybrid_hessian_contract_accepts_exact_abi_and_global_multipliers():
    native = _hessian("native")
    candidate = _hessian("candidate")
    report = _MODULE.audit(native, candidate, selected_lambda_rows=np.array([0, 1]), samples=2, repeats=1)
    assert report["passed"]
    assert report["max_abs_error"] == 0.0
    assert report["max_abs_selected_lambda_response_error"] == 0.0


def test_hybrid_hessian_contract_rejects_wrong_multiplier_response():
    native = _hessian("native")
    candidate = _hessian("candidate", offset=0.1)
    report = _MODULE.audit(native, candidate, selected_lambda_rows=[0, 1], samples=1, repeats=1)
    assert not report["passed"]
