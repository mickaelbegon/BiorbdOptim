"""Experimental, signature-preserving CasADi 3.8 IPOPT callback transforms."""

from time import perf_counter

import casadi


TRANSFORM_PASSES = (("simplify", "cse", "ref_count", "const_folding"),)


def require_function_transform():
    """Fail explicitly rather than silently ignoring an unavailable optimization."""
    version = tuple(int(part) for part in casadi.__version__.split(".")[:2])
    if version < (3, 8) or not hasattr(casadi.Function, "transform"):
        raise RuntimeError("IPOPT function_transform requires CasADi >= 3.8 with Function.transform support")


def transformed_nlpsol(nlp, options):
    """Transform the callbacks selected by IPOPT, returning solver and diagnostics.

    The explicit pipeline intentionally excludes ``empty_inputs``: IPOPT's
    empty parameter input and the derivative sparsities are part of its ABI.
    CasADi's function cache installs primal and derivative callbacks alike.
    C code generation must use the returned solver, while native reload must
    use the original options so these VM functions do not override compiled C.
    """
    require_function_transform()
    started = perf_counter()
    reference = casadi.nlpsol("nlpsol", "ipopt", nlp, options)
    callbacks = dict(options.get("cache", {}))
    rows = {}
    for name in reference.get_function():
        original = reference.get_function(name)
        tick = perf_counter()
        transformed = original.transform(name, [list(p) for p in TRANSFORM_PASSES], {})
        same_signature = (
            original.name_in() == transformed.name_in()
            and original.name_out() == transformed.name_out()
            and all(original.sparsity_in(i) == transformed.sparsity_in(i) for i in range(original.n_in()))
            and all(original.sparsity_out(i) == transformed.sparsity_out(i) for i in range(original.n_out()))
        )
        if not same_signature:
            raise RuntimeError(f"Function.transform changed the IPOPT callback signature or sparsity: {name}")
        callbacks[name] = transformed
        rows[name] = {
            "transform_seconds": perf_counter() - tick,
            "instructions_before": original.n_instructions(),
            "instructions_after": transformed.n_instructions(),
        }
    solver = casadi.nlpsol("nlpsol", "ipopt", nlp, {**options, "cache": callbacks})
    return solver, {
        "passes": [list(p) for p in TRANSFORM_PASSES],
        "preparation_seconds": perf_counter() - started,
        "callbacks": rows,
    }
