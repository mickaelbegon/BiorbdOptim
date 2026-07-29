from copy import deepcopy
from time import perf_counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import linalg
from casadi import SX, vertcat, Function
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from acados_template.utils import status_to_str

from .solver_interface import SolverInterface
from ..interfaces import Solver
from ..misc.enums import Node, SolverType, PhaseDynamics
from ..limits.objective_functions import ObjectiveFunction, ObjectiveFcn
from ..limits.path_conditions import Bounds
from ..misc.enums import InterpolationType
from ..optimization.solution.solution import Solution


from ..misc.parameters_types import (
    Str,
    Bool,
    AnyListorDict,
)

_ACADOS_TIMING_FIELDS = (
    "time_tot",
    "time_lin",
    "time_sim",
    "time_sim_ad",
    "time_sim_la",
    "time_qp",
    "time_qp_solver_call",
    "time_qp_xcond",
    "time_glob",
    "time_qpscaling",
    "time_reg",
    "time_preparation",
    "time_feedback",
)

_ACADOS_WARM_START_FIELDS = ("x", "u", "pi", "lam", "sl", "su")
_ACADOS_SOLVER_STATE_FORMAT_VERSION = 1


def _get_acados_runtime_parameters(nlp) -> np.ndarray:
    """Flatten Bioptim numerical time series in the same order as their symbolic variables."""

    if nlp.numerical_data_timeseries is None:
        return np.zeros((0, nlp.ns + 1))

    runtime_parameters = []
    for key, values in nlp.numerical_data_timeseries.items():
        if values.ndim != 3 or values.shape[2] != nlp.ns + 1:
            raise ValueError(
                f"numerical_data_timeseries['{key}'] must have shape (n_values, n_components, {nlp.ns + 1}), "
                f"got {values.shape}."
            )
        for component_index in range(values.shape[1]):
            runtime_parameters.append(np.asarray(values[:, component_index, :], dtype=float))

    return np.vstack(runtime_parameters) if runtime_parameters else np.zeros((0, nlp.ns + 1))


def _safe_acados_stat(acados_solver: AcadosOcpSolver, field: Str, errors: dict) -> int | float | np.ndarray | None:
    """Read a diagnostic value without masking an otherwise usable solve."""

    try:
        value = acados_solver.get_stats(field)
    except Exception as exc:  # Diagnostics must remain available after partial solver failures.
        errors[field] = f"{type(exc).__name__}: {exc}"
        return None

    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _statistics_solver_details(statistics: np.ndarray | None, nlp_solver_type: Str) -> dict:
    """Extract solver-independent QP histories from Acados' solver-specific table."""

    details = {
        "qp_status": None,
        "qp_iterations": None,
        "qp_iterations_per_solve": None,
        "step_sizes": None,
    }
    if statistics is None or statistics.ndim != 2 or statistics.size == 0:
        return details

    if nlp_solver_type == "SQP" and statistics.shape[0] >= 8:
        details["qp_status"] = statistics[5, :].copy()
        details["qp_iterations"] = statistics[6, :].copy()
        details["step_sizes"] = statistics[7, :].copy()
    elif nlp_solver_type == "SQP_RTI" and statistics.shape[0] >= 3:
        details["qp_status"] = statistics[1, :].copy()
        details["qp_iterations"] = statistics[2, :].copy()
    elif nlp_solver_type == "DDP" and statistics.shape[0] >= 8:
        details["qp_status"] = statistics[5, :].copy()
        details["qp_iterations"] = statistics[6, :].copy()
        details["step_sizes"] = statistics[7, :].copy()
    elif nlp_solver_type == "SQP_WITH_FEASIBLE_QP" and statistics.shape[0] >= 14:
        details["qp_status"] = statistics[[5, 7, 9], :].T.copy()
        details["qp_iterations_per_solve"] = statistics[[6, 8, 10], :].T.copy()
        details["qp_iterations"] = np.sum(details["qp_iterations_per_solve"], axis=1)
        details["step_sizes"] = statistics[11, :].copy()

    return details


def _collect_acados_diagnostics(
    acados_solver: AcadosOcpSolver,
    status: int | None,
    nlp_solver_type: Str,
    qp_solver: Str,
) -> dict:
    """Create an immutable snapshot of the public diagnostics exposed by Acados v0.5.5."""

    errors = {}
    raw_statistics = _safe_acados_stat(acados_solver, "statistics", errors)
    raw_statistics = np.asarray(raw_statistics, dtype=float) if raw_statistics is not None else None
    solver_details = _statistics_solver_details(raw_statistics, nlp_solver_type)

    try:
        residual_values = np.asarray(acados_solver.get_residuals(recompute=False), dtype=float).reshape(-1).copy()
    except Exception as exc:  # Diagnostics must remain available after partial solver failures.
        errors["residuals"] = f"{type(exc).__name__}: {exc}"
        residual_values = np.array([], dtype=float)

    residuals = None
    if residual_values.size >= 4:
        residuals = dict(
            zip(
                ("stationarity", "dynamics", "inequality", "complementarity"),
                (float(value) for value in residual_values[:4]),
            )
        )

    timings = {
        field: value
        for field in _ACADOS_TIMING_FIELDS
        if (value := _safe_acados_stat(acados_solver, field, errors)) is not None
    }
    status_value = -1 if status is None else int(status)
    diagnostics = {
        "status": status,
        "status_label": status_to_str(status_value),
        "successful": status == 0,
        "nlp_solver_type": nlp_solver_type,
        "qp_solver": qp_solver,
        "residuals": residuals,
        "nlp_iterations": _safe_acados_stat(acados_solver, "nlp_iter", errors),
        "sqp_iterations": _safe_acados_stat(acados_solver, "sqp_iter", errors),
        "qp_scaling_status": _safe_acados_stat(acados_solver, "qpscaling_status", errors),
        "timings": timings,
        "raw_statistics": raw_statistics,
        **solver_details,
    }
    if errors:
        diagnostics["unavailable_statistics"] = errors
    return diagnostics


def _configure_acados_codegen(acados_ocp: AcadosOcp, solver_options: Solver.ACADOS) -> None:
    """Configure code generation with the Acados v0.5.5 API."""

    if not hasattr(acados_ocp, "code_gen_options"):
        raise RuntimeError(
            "This version of bioptim requires Acados v0.5.5 or newer. "
            "Please reinstall Acados using the scripts from bioptim/external."
        )

    code_gen_options = acados_ocp.code_gen_options
    if solver_options.acados_dir:
        acados_dir = Path(solver_options.acados_dir).expanduser().resolve()
        code_gen_options.acados_include_path = str(acados_dir / "include")
        code_gen_options.acados_lib_path = str(acados_dir / "lib")

    code_gen_options.code_export_directory = solver_options.c_generated_code_path
    code_gen_options.json_file = "acados_ocp.json"


class AcadosInterface(SolverInterface):
    """
    The ACADOS solver interface

    Attributes
    ----------
    acados_ocp: AcadosOcp
        The current AcadosOcp reference
    acados_model: AcadosModel
        The current AcadosModel reference
    lagrange_costs: SX
        The lagrange cost function
    mayer_costs: SX
        The mayer cost function
    y_ref = list[np.ndarray]
        The lagrange targets
    y_ref_end = list[np.ndarray]
        The mayer targets
    params = dict
        All the parameters to optimize
    W: np.ndarray
        The Lagrange weights
    W_e: np.ndarray
        The Mayer weights
    status: int
        The status of the optimization
    all_constr: SX
        All the Lagrange constraints
    end_constr: SX
        All the Mayer constraints
    all_g_bounds = Bounds
        All the Lagrange bounds on the variables
    end_g_bounds = Bounds
        All the Mayer bounds on the variables
    x_bound_max = np.ndarray
        All the bounds max
    x_bound_min = np.ndarray
        All the bounds min
    Vu: np.ndarray
        The control objective functions
    Vx: np.ndarray
        The Lagrange state objective functions
    Vxe: np.ndarray
        The Mayer state objective functions
    opts: ACADOS
        Options of Acados from ACADOS
    Methods
    -------
    __acados_export_model(self, ocp: OptimalControlProgram)
        Creating a generic ACADOS model
    __prepare_acados(self, ocp: OptimalControlProgram)
        Set some important ACADOS variables
    __set_constr_type(self, constr_type: str = "BGH")
        Set the type of constraints
    __set_constraints(self, ocp: OptimalControlProgram)
        Set the constraints from the ocp
    __set_cost_type(self, cost_type: str = "NONLINEAR_LS")
        Set the type of cost functions
    __set_costs(self, ocp: OptimalControlProgram)
        Set the cost functions from ocp
    __update_solver(self)
        Update the ACADOS solver to new values
    get_optimized_value(self) -> list[dict] | dict
        Get the previously optimized solution
    get_diagnostics(self) -> dict
        Get a snapshot of the diagnostics from the previous solver call
    solve(self) -> "AcadosInterface"
        Solve the prepared ocp
    """

    def __init__(self, ocp, solver_options: Solver.ACADOS = None):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        solver_options: ACADOS
            The options to pass to the solver
        """

        if not isinstance(ocp.cx(), SX):
            raise RuntimeError("CasADi graph must be SX to be solved with ACADOS. Please set use_sx to True in OCP")

        if ocp.nlp[0].phase_dynamics != PhaseDynamics.SHARED_DURING_THE_PHASE:
            raise RuntimeError("ACADOS necessitate phase_dynamics==PhaseDynamics.SHARED_DURING_THE_PHASE")

        if ocp.nlp[0].algebraic_states.cx_start.shape[0] != 0:
            raise RuntimeError("ACADOS does not support algebraic states yet")

        super().__init__(ocp)

        # solver_options = solver_options.__dict__
        if solver_options is None:
            solver_options = Solver.ACADOS()
        self.opts = solver_options

        self.acados_ocp = AcadosOcp()
        _configure_acados_codegen(self.acados_ocp, solver_options)
        self.acados_model = AcadosModel()

        self.__set_cost_type(solver_options.cost_type)
        self.__set_constr_type(solver_options.constr_type)

        self.lagrange_costs = SX()
        self.mayer_costs_e = SX()
        self.mayer_costs = SX()
        self.y_ref = []
        self.y_ref_end = []
        self.y_ref_start = []
        self.nparams = 0
        self.__acados_export_model(ocp)
        self.__prepare_acados(ocp)
        self.ocp_solver = None
        self.W = np.zeros((0, 0))
        self.W_e = np.zeros((0, 0))
        self.W_0 = np.zeros((0, 0))
        self.status = None
        self.out = {}
        self.real_time_to_optimize = -1
        self._runtime_parameter_values = None
        self._warm_start_solver_state = None

        self.all_constr = None
        self.end_constr = SX()
        self.all_g_bounds = Bounds(None, interpolation=InterpolationType.CONSTANT)
        self.end_g_bounds = Bounds(None, interpolation=InterpolationType.CONSTANT)
        self.x_bound_max = np.ndarray((self.acados_ocp.dims.nx, 3))
        self.x_bound_min = np.ndarray((self.acados_ocp.dims.nx, 3))
        self.Vu = np.array([], dtype=np.int64).reshape(0, ocp.nlp[0].controls.shape)
        self.Vx = np.array([], dtype=np.int64).reshape(0, ocp.nlp[0].states.shape)
        self.Vxe = np.array([], dtype=np.int64).reshape(0, ocp.nlp[0].states.shape)
        self.Vx0 = np.array([], dtype=np.int64).reshape(0, ocp.nlp[0].states.shape)

    def __acados_export_model(self, ocp):
        """
        Creating a generic ACADOS model

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram

        """

        if ocp.n_phases > 1:
            raise NotImplementedError("More than 1 phase is not implemented yet with ACADOS backend")

        # Declare model variables
        t = ocp.nlp[0].time_cx
        x = ocp.nlp[0].states.cx_start
        x_sym = ocp.nlp[0].states.scaled.cx_start
        u = ocp.nlp[0].controls.cx_start
        u_sym = ocp.nlp[0].controls.scaled.cx_start
        p = ocp.nlp[0].parameters.scaled.cx
        p_sym = ocp.nlp[0].parameters.scaled.cx
        a = ocp.nlp[0].algebraic_states.cx_start
        a_sym = ocp.nlp[0].algebraic_states.scaled.cx_start
        d = ocp.nlp[0].numerical_timeseries.cx_start

        if ocp.parameters:
            for key in ocp.parameters:
                if str(ocp.parameters[key].cx)[:11] == f"time_phase_":
                    raise RuntimeError("Time constraint not implemented yet with Acados.")
        if a_sym.shape[0] != 0:
            raise RuntimeError("Algebraic states not implemented yet with Acados.")

        self.nparams = ocp.nlp[0].parameters.shape

        x_sym = vertcat(p_sym, x_sym)
        x_dot_sym = SX.sym("x_dot", x_sym.shape[0], x_sym.shape[1])

        f_expl = vertcat([0] * self.nparams, ocp.nlp[0].dynamics_func(t, x, u, p, a, d))
        f_impl = x_dot_sym - f_expl

        self.acados_model.f_impl_expr = f_impl
        self.acados_model.f_expl_expr = f_expl
        self.acados_model.x = x_sym
        self.acados_model.xdot = x_dot_sym
        self.acados_model.u = u_sym
        self.acados_model.p = d
        self.acados_model.con_h_expr_0 = np.zeros((0, 0))
        self.acados_model.con_h_expr = np.zeros((0, 0))
        self.acados_model.con_h_expr_e = np.zeros((0, 0))
        if not self.opts.acados_model_name:
            if self.opts.check_reuse_possible:
                raise RuntimeError(
                    "Acados code reuse requires a stable model name. "
                    "Please call Solver.ACADOS.set_acados_model_name(...)."
                )
            elif self.opts.c_compile:
                now = datetime.now()  # current date and time
                self.acados_model.name = f"model_{now.strftime('%Y_%m_%d_%H%M%S%f')[:-4]}"
            else:
                raise RuntimeError(
                    "If not compiling the library you must provide the name of the model" " you want to use."
                )
        else:
            self.acados_model.name = self.opts.acados_model_name

    def __prepare_acados(self, ocp):
        """
        Set some important ACADOS variables

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        # set model
        self.acados_ocp.model = self.acados_model

        # set time
        tf_init = ocp.dt_parameter_initial_guess.init[0, 0]
        tf = float(Function("tf", [ocp.nlp[0].dt], [ocp.nlp[0].tf])(tf_init))
        self.acados_ocp.solver_options.tf = tf

        # set dimensions
        self.acados_ocp.dims.nx = ocp.nlp[0].parameters.shape + ocp.nlp[0].states.shape
        self.acados_ocp.dims.nu = ocp.nlp[0].controls.shape
        self.acados_ocp.solver_options.N_horizon = ocp.nlp[0].ns
        runtime_parameters = _get_acados_runtime_parameters(ocp.nlp[0])
        self.acados_ocp.parameter_values = runtime_parameters[:, 0].copy()

    def __set_constr_type(self, constr_type: Str = "BGH") -> None:
        """
        Set the type of constraints

        Parameters
        ----------
        constr_type: str
            The requested type of constraints
        """

        self.acados_ocp.constraints.constr_type = constr_type
        self.acados_ocp.constraints.constr_type_e = constr_type

    def __set_constraints(self, ocp) -> None:
        """
        Set the constraints from the ocp

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        # constraints handling in self.acados_ocp
        if ocp.nlp[0].x_bounds.type != InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            raise NotImplementedError(
                "ACADOS must declare an InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT " "for the x_bounds"
            )
        if ocp.nlp[0].u_bounds.type != InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            raise NotImplementedError(
                "ACADOS must declare an InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT " "for the u_bounds"
            )

        for key in ocp.nlp[0].controls.keys():
            if not np.all(np.all(ocp.nlp[0].u_bounds[key].min.T == ocp.nlp[0].u_bounds[key].min.T[0, :], axis=0)):
                raise NotImplementedError("u_bounds min must be the same at each shooting point with ACADOS")
            if not np.all(np.all(ocp.nlp[0].u_bounds[key].max.T == ocp.nlp[0].u_bounds[key].max.T[0, :], axis=0)):
                raise NotImplementedError("u_bounds max must be the same at each shooting point with ACADOS")

            if (
                not np.isfinite(ocp.nlp[0].u_bounds[key].min).all()
                or not np.isfinite(ocp.nlp[0].u_bounds[key].max).all()
            ):
                raise NotImplementedError(
                    "u_bounds and x_bounds cannot be set to infinity in ACADOS. Consider changing it "
                    "to a big value instead."
                )

        for key in ocp.nlp[0].states.keys():
            if (
                not np.isfinite(ocp.nlp[0].x_bounds[key].min).all()
                or not np.isfinite(ocp.nlp[0].x_bounds[key].max).all()
            ):
                raise NotImplementedError(
                    "u_bounds and x_bounds cannot be set to infinity in ACADOS. Consider changing it "
                    "to a big value instead."
                )

        self.all_constr = SX()
        self.end_constr = SX()
        # TODO:change for more node flexibility on bounds
        self.all_g_bounds = Bounds(None, interpolation=InterpolationType.CONSTANT)
        self.end_g_bounds = Bounds(None, interpolation=InterpolationType.CONSTANT)
        for i, nlp in enumerate(ocp.nlp):
            t = nlp.time_cx
            dt = nlp.dt
            x = nlp.states.cx_start
            u = nlp.controls.cx_start
            p = nlp.parameters.scaled.cx
            a = nlp.algebraic_states.cx_start
            d = nlp.numerical_timeseries.cx

            for g, G in enumerate(nlp.g):
                if not G:
                    continue

                if G.node[0] == Node.ALL or G.node[0] == Node.ALL_SHOOTING:
                    x_tp = x
                    u_tp = u
                    if x.shape[0] * 2 == G.function[0].size_in("x")[0]:
                        x_tp = vertcat(x_tp, x_tp)
                    if u.shape[0] * 2 == G.function[0].size_in("u")[0]:
                        u_tp = vertcat(u_tp, u_tp)

                    self.all_constr = vertcat(self.all_constr, G.function[0](t, dt, x_tp, u_tp, p, a, d))
                    self.all_g_bounds.concatenate(G.bounds)
                    if G.node[0] == Node.ALL:
                        self.end_constr = vertcat(self.end_constr, G.function[0](t, dt, x_tp, u_tp, p, a, d))
                        self.end_g_bounds.concatenate(G.bounds)

                elif G.node[0] == Node.END:
                    x_tp = x
                    u_tp = u
                    if x.shape[0] * 2 == G.function[-1].size_in("x")[0]:
                        x_tp = vertcat(x_tp, x_tp)
                    if u.shape[0] * 2 == G.function[-1].size_in("u")[0]:
                        u_tp = vertcat(u_tp, u_tp)

                    self.end_constr = vertcat(self.end_constr, G.function[-1](t, dt, x_tp, u_tp, p, a, d))
                    self.end_g_bounds.concatenate(G.bounds)

                else:
                    raise RuntimeError(
                        "Except for states and controls, Acados solver only handles constraints on last or all nodes."
                    )

        self.acados_model.con_h_expr_0 = self.all_constr
        self.acados_model.con_h_expr = self.all_constr
        self.acados_model.con_h_expr_e = self.end_constr

        # setup state constraints
        # TODO replace all these np.concatenate by proper bound and initial_guess classes
        self.x_bound_max = np.ndarray((self.acados_ocp.dims.nx, 3))
        self.x_bound_min = np.ndarray((self.acados_ocp.dims.nx, 3))
        param_bounds_max = []
        param_bounds_min = []

        for key in ocp.parameter_bounds.keys():
            param_bounds_scale = ocp.parameter_bounds[key].scale(ocp.parameters[key].scaling.scaling)
            param_bounds_max = np.concatenate((param_bounds_max, param_bounds_scale.max[:, 0]))
            param_bounds_min = np.concatenate((param_bounds_min, param_bounds_scale.min[:, 0]))

        if self.nparams > 0:
            self.x_bound_max[: self.nparams, :] = np.repeat(param_bounds_max[:, np.newaxis], 3, axis=1)
            self.x_bound_min[: self.nparams, :] = np.repeat(param_bounds_min[:, np.newaxis], 3, axis=1)

        for key in ocp.nlp[0].states.keys():
            x_tp = ocp.nlp[0].x_bounds[key].scale(ocp.nlp[0].x_scaling[key].scaling)
            index = [i + self.nparams for i in ocp.nlp[0].states[key].index]
            for i in range(3):
                self.x_bound_max[index, i] = x_tp.max[:, i]
                self.x_bound_min[index, i] = x_tp.min[:, i]

        # setup control constraints
        u_bounds_max = np.ndarray((self.acados_ocp.dims.nu, 1))
        u_bounds_min = np.ndarray((self.acados_ocp.dims.nu, 1))
        for key in ocp.nlp[0].controls.keys():
            u_tp = ocp.nlp[0].u_bounds[key].scale(ocp.nlp[0].u_scaling[key].scaling)
            index = ocp.nlp[0].controls[key].index
            u_bounds_max[index, 0] = np.array(u_tp.max[:, 0])
            u_bounds_min[index, 0] = np.array(u_tp.min[:, 0])

        self.acados_ocp.constraints.lbu = u_bounds_max
        self.acados_ocp.constraints.ubu = u_bounds_min
        self.acados_ocp.constraints.idxbu = np.array(range(self.acados_ocp.dims.nu))
        self.acados_ocp.dims.nbu = self.acados_ocp.dims.nu

        # initial state constraints
        self.acados_ocp.constraints.Jbx_0 = np.eye(self.acados_ocp.dims.nx)
        self.acados_ocp.constraints.lbx_0 = self.x_bound_min[:, 0]
        self.acados_ocp.constraints.ubx_0 = self.x_bound_max[:, 0]
        self.acados_ocp.constraints.idxbx_0 = np.array(range(self.acados_ocp.dims.nx))
        self.acados_ocp.dims.nbx_0 = self.acados_ocp.dims.nx

        # setup path state constraints
        self.acados_ocp.constraints.Jbx = np.eye(self.acados_ocp.dims.nx)
        self.acados_ocp.constraints.lbx = self.x_bound_min[:, 1]
        self.acados_ocp.constraints.ubx = self.x_bound_max[:, 1]
        self.acados_ocp.constraints.idxbx = np.array(range(self.acados_ocp.dims.nx))
        self.acados_ocp.dims.nbx = self.acados_ocp.dims.nx

        # setup terminal state constraints
        self.acados_ocp.constraints.Jbx_e = np.eye(self.acados_ocp.dims.nx)
        self.acados_ocp.constraints.lbx_e = self.x_bound_min[:, -1]
        self.acados_ocp.constraints.ubx_e = self.x_bound_max[:, -1]
        self.acados_ocp.constraints.idxbx_e = np.array(range(self.acados_ocp.dims.nx))
        self.acados_ocp.dims.nbx_e = self.acados_ocp.dims.nx

        # setup algebraic constraint
        self.acados_ocp.constraints.lh_0 = np.array(self.all_g_bounds.min[:, 0])
        self.acados_ocp.constraints.uh_0 = np.array(self.all_g_bounds.max[:, 0])
        self.acados_ocp.constraints.lh = np.array(self.all_g_bounds.min[:, 0])
        self.acados_ocp.constraints.uh = np.array(self.all_g_bounds.max[:, 0])
        self.acados_ocp.constraints.lh_e = np.array(self.end_g_bounds.min[:, 0])
        self.acados_ocp.constraints.uh_e = np.array(self.end_g_bounds.max[:, 0])

    def __set_cost_type(self, cost_type: Str = "NONLINEAR_LS") -> None:
        """
        Set the type of cost functions

        Parameters
        ----------
        cost_type: str
            The type of cost function
        """

        self.acados_ocp.cost.cost_type = cost_type
        self.acados_ocp.cost.cost_type_e = cost_type
        self.acados_ocp.cost.cost_type_0 = cost_type

    def __set_costs(self, ocp) -> None:
        """
        Set the cost functions from ocp

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        def add_linear_ls_lagrange(acados, objectives):
            def add_objective(n_variables, is_state):
                # Sanity check for weightings
                if objectives.weight.type != InterpolationType.CONSTANT:
                    raise RuntimeError("Lagrange objective weight must be InterpolationType.CONSTANT.")

                v_var = np.zeros(n_variables)
                var_type = acados.ocp.nlp[0].states if is_state else acados.ocp.nlp[0].controls
                rows = objectives.rows + var_type[objectives.extra_parameters["key"]].index[0]
                v_var[rows] = 1.0
                if is_state:
                    acados.Vx = np.vstack((acados.Vx, np.diag(v_var)))
                    acados.Vu = np.vstack((acados.Vu, np.zeros((n_states, n_controls))))
                else:
                    acados.Vx = np.vstack((acados.Vx, np.zeros((n_controls, n_states))))
                    acados.Vu = np.vstack((acados.Vu, np.diag(v_var)))
                acados.W = linalg.block_diag(acados.W, np.diag(objectives.weight.evaluate_at(0, n_variables)))

                node_idx = objectives.node_idx[:-1] if objectives.node[0] == Node.ALL else objectives.node_idx

                y_ref = [np.zeros((n_states if is_state else n_controls, 1)) for _ in node_idx]
                if objectives.target is not None:
                    for idx in node_idx:
                        y_ref[idx][rows] = objectives.target[..., idx].T.reshape((-1, 1))
                acados.y_ref.append(y_ref)

            if objectives.type in allowed_control_objectives:
                add_objective(n_controls, False)
            elif objectives.type in allowed_state_objectives:
                add_objective(n_states, True)
            else:
                raise RuntimeError(
                    f"{objectives[0]['objective'].type.name} is an incompatible objective term with LINEAR_LS cost type"
                )

        def add_linear_ls_mayer(acados, objectives):
            def add_objective(n_variables, is_state):
                def _adjust_dim():
                    v_var = np.zeros(n_variables)
                    var_type = acados.ocp.nlp[0].states if is_state else acados.ocp.nlp[0].controls
                    rows = objectives.rows + var_type[objectives.extra_parameters["key"]].index[0]
                    v_var[rows] = 1.0
                    return v_var, rows

                if objectives.node[0] not in [Node.INTERMEDIATES, Node.PENULTIMATE, Node.END]:
                    v_var, rows = _adjust_dim()
                    if is_state:
                        acados.Vx0 = np.vstack((acados.Vx0, np.diag(v_var)))
                        acados.Vu0 = np.vstack((acados.Vu0, np.zeros((n_states, n_controls))))
                    else:
                        acados.Vx0 = np.vstack((acados.Vx0, np.zeros((n_controls, n_states))))
                        acados.Vu0 = np.vstack((acados.Vu0, np.diag(v_var)))
                    acados.W_0 = linalg.block_diag(acados.W_0, np.diag(objectives.weight.evaluate_at(0, n_variables)))
                    y_ref_start = np.zeros((n_variables, 1))
                    if objectives.target is not None:
                        y_ref_start[rows] = objectives.target[..., 0].T.reshape((-1, 1))
                    acados.y_ref_start.append(y_ref_start)

                if objectives.node[0] in [Node.END, Node.ALL]:
                    v_var, rows = _adjust_dim()
                    if not is_state:
                        raise RuntimeError("Mayer objective at final node for controls is not defined.")
                    acados.Vxe = np.vstack((acados.Vxe, np.diag(v_var)))
                    acados.W_e = linalg.block_diag(acados.W_e, np.diag(objectives.weight.evaluate_at(0, n_states)))
                    y_ref_end = np.zeros((n_states, 1))
                    if objectives.target is not None:
                        y_ref_end[rows] = objectives.target[..., -1].T.reshape((-1, 1))
                    acados.y_ref_end.append(y_ref_end)

            if objectives.type in allowed_control_objectives:
                add_objective(n_controls, False)
            elif objectives.type in allowed_state_objectives:
                add_objective(n_states, True)
            else:
                raise RuntimeError(f"{objectives.type.name} is an incompatible objective term with LINEAR_LS cost type")

        def add_nonlinear_ls_lagrange(acados, objectives, t, dt, x, u, p, a, d):
            if objectives.function[0].size_in("x")[0] == x.shape[0] * 2:
                x = vertcat(x, x)
            if objectives.function[0].size_in("u")[0] == u.shape[0] * 2:
                u = vertcat(u, u)

            acados.lagrange_costs = vertcat(
                acados.lagrange_costs, objectives.function[0](t, dt, x, u, p, a, d).reshape((-1, 1))
            )
            acados.W = linalg.block_diag(
                acados.W, np.diag(objectives.weight.evaluate_at(0, objectives.function[0].numel_out()))
            )

            node_idx = objectives.node_idx[:-1] if objectives.node[0] == Node.ALL else objectives.node_idx
            if objectives.target is not None:
                acados.y_ref.append([objectives.target[..., idx].T.reshape((-1, 1)) for idx in node_idx])
            else:
                acados.y_ref.append([np.zeros((objectives.function[0].numel_out(), 1)) for _ in node_idx])

        def add_nonlinear_ls_mayer(acados, objectives, t, dt, x, u, p, a, d, node=None):
            if objectives.node[0] not in [Node.INTERMEDIATES, Node.PENULTIMATE, Node.END]:
                acados.W_0 = linalg.block_diag(
                    acados.W_0, np.diag(objectives.weight.evaluate_at(0, objectives.function[0].numel_out()))
                )

                x_tp = x
                u_tp = u
                if objectives.function[0].size_in("x")[0] == x_tp.shape[0] * 2:
                    x_tp = vertcat(x_tp, x_tp)
                if objectives.function[0].size_in("u")[0] == u_tp.shape[0] * 2:
                    u_tp = vertcat(u_tp, u_tp)

                x_tp = x_tp if objectives.function[0].size_in("x") != (0, 0) else []
                u_tp = u_tp if objectives.function[0].size_in("u") != (0, 0) else []

                acados.mayer_costs = vertcat(
                    acados.mayer_costs, objectives.function[0](t, dt, x_tp, u_tp, p, a, d).reshape((-1, 1))
                )

                if objectives.target is not None:
                    acados.y_ref_start.append(objectives.target[..., 0].T.reshape((-1, 1)))
                else:
                    acados.y_ref_start.append(np.zeros((objectives.function[0].numel_out(), 1)))

            if objectives.node[0] in [Node.END, Node.ALL]:
                acados.W_e = linalg.block_diag(
                    acados.W_e, np.diag(objectives.weight.evaluate_at(0, objectives.function[-1].numel_out()))
                )
                x_tp = x
                u_tp = u
                if objectives.function[-1].size_in("x")[0] == x_tp.shape[0] * 2:
                    x_tp = vertcat(x_tp, x_tp)
                if objectives.function[-1].size_in("u")[0] == u_tp.shape[0] * 2:
                    u_tp = vertcat(u_tp, u_tp)

                x_tp = x_tp if objectives.function[-1].size_in("x") != (0, 0) else []
                u_tp = u_tp if objectives.function[-1].size_in("u") != (0, 0) else []

                acados.mayer_costs_e = vertcat(
                    acados.mayer_costs_e, objectives.function[-1](t, dt, x_tp, u_tp, p, a, d).reshape((-1, 1))
                )

                if objectives.target is not None:
                    acados.y_ref_end.append(objectives.target[..., -1].T.reshape((-1, 1)))
                else:
                    acados.y_ref_end.append(np.zeros((objectives.function[-1].numel_out(), 1)))

        if ocp.n_phases != 1:
            raise NotImplementedError("ACADOS with more than one phase is not implemented yet.")
        # costs handling in self.acados_ocp
        self.y_ref = []
        self.y_ref_end = []
        self.y_ref_start = []
        self.lagrange_costs = SX()
        self.mayer_costs_e = SX()
        self.mayer_costs = SX()
        self.W = np.zeros((0, 0))
        self.W_e = np.zeros((0, 0))
        self.W_0 = np.zeros((0, 0))
        allowed_control_objectives = [ObjectiveFcn.Lagrange.MINIMIZE_CONTROL]
        allowed_state_objectives = [ObjectiveFcn.Lagrange.MINIMIZE_STATE, ObjectiveFcn.Mayer.TRACK_STATE]

        if self.acados_ocp.cost.cost_type == "LINEAR_LS":
            n_states = ocp.nlp[0].states.shape
            n_controls = ocp.nlp[0].controls.shape
            self.Vu = np.array([], dtype=np.int64).reshape(0, n_controls)
            self.Vx = np.array([], dtype=np.int64).reshape(0, n_states)
            self.Vxe = np.array([], dtype=np.int64).reshape(0, n_states)
            self.Vx0 = np.array([], dtype=np.int64).reshape(0, n_states)
            self.Vu0 = np.array([], dtype=np.int64).reshape(0, n_controls)
            for i in range(ocp.n_phases):
                for J in ocp.nlp[i].J:
                    if not J:
                        continue

                    if J.multi_thread:
                        raise RuntimeError(
                            f"The objective function {J.name} was declared with multi_thread=True, "
                            f"but this is not possible to multi_thread objective function with ACADOS"
                        )

                    if J.type.get_type() == ObjectiveFunction.LagrangeFunction:
                        add_linear_ls_lagrange(self, J)

                        # Deal with first and last node
                        add_linear_ls_mayer(self, J)

                    elif J.type.get_type() == ObjectiveFunction.MayerFunction:
                        add_linear_ls_mayer(self, J)

                    else:
                        raise RuntimeError("The objective function is not Lagrange nor Mayer.")

                if self.nparams:
                    raise RuntimeError("Params not yet handled with LINEAR_LS cost type")

            # Set costs
            self.acados_ocp.cost.Vx = self.Vx if self.Vx.shape[0] else np.zeros((0, 0))
            self.acados_ocp.cost.Vu = self.Vu if self.Vu.shape[0] else np.zeros((0, 0))
            self.acados_ocp.cost.Vu_0 = self.Vu0 if self.Vu0.shape[0] else np.zeros((0, 0))
            self.acados_ocp.cost.Vx_e = self.Vxe if self.Vxe.shape[0] else np.zeros((0, 0))
            self.acados_ocp.cost.Vx_0 = self.Vx0 if self.Vx0.shape[0] else np.zeros((0, 0))

            # Set dimensions
            self.acados_ocp.dims.ny = sum([len(data[0]) for data in self.y_ref])
            self.acados_ocp.dims.ny_e = sum([len(data) for data in self.y_ref_end])
            self.acados_ocp.dims.ny_0 = sum([len(data) for data in self.y_ref_start])

            # Set weight
            self.acados_ocp.cost.W = self.W
            self.acados_ocp.cost.W_e = self.W_e
            self.acados_ocp.cost.W_0 = self.W_0

            # Set target shape
            self.acados_ocp.cost.yref = np.zeros((self.acados_ocp.dims.ny,))
            self.acados_ocp.cost.yref_e = np.zeros((self.acados_ocp.dims.ny_e,))
            self.acados_ocp.cost.yref_0 = np.zeros((self.acados_ocp.dims.ny_0,))

        elif self.acados_ocp.cost.cost_type == "NONLINEAR_LS":
            for i, nlp in enumerate(ocp.nlp):
                for j, J in enumerate(nlp.J):
                    if not J:
                        continue

                    if J.multi_thread:
                        raise RuntimeError(
                            f"The objective function {J.name} was declared with multi_thread=True, "
                            f"but this is not possible to multi_thread objective function with ACADOS"
                        )

                    if J.type.get_type() == ObjectiveFunction.LagrangeFunction:
                        add_nonlinear_ls_lagrange(
                            self,
                            J,
                            nlp.time_cx,
                            nlp.dt,
                            nlp.states.scaled.cx_start,
                            nlp.controls.scaled.cx_start,
                            nlp.parameters.scaled.cx,
                            nlp.algebraic_states.scaled.cx_start,
                            nlp.numerical_timeseries.cx,
                        )

                        # Deal with first and last node
                        add_nonlinear_ls_mayer(
                            self,
                            J,
                            nlp.time_cx,
                            nlp.dt,
                            nlp.states.scaled.cx_start,
                            nlp.controls.scaled.cx_start,
                            nlp.parameters.scaled.cx,
                            nlp.algebraic_states.scaled.cx_start,
                            nlp.numerical_timeseries.cx,
                        )

                    elif J.type.get_type() == ObjectiveFunction.MayerFunction:
                        add_nonlinear_ls_mayer(
                            self,
                            J,
                            nlp.time_cx,
                            nlp.dt,
                            nlp.states.scaled.cx_start,
                            nlp.controls.scaled.cx_start,
                            nlp.parameters.scaled.cx,
                            nlp.algebraic_states.scaled.cx_start,
                            nlp.numerical_timeseries.cx,
                        )
                    else:
                        raise RuntimeError("The objective function is not Lagrange nor Mayer.")

            # parameter as mayer function
            # IMPORTANT: it is considered that only parameters are stored in ocp.objectives, for now.
            if self.nparams:
                nlp = ocp.nlp[0]  # Assume 1 phase
                for j, J in enumerate(ocp.J):
                    J.node = [Node.END]
                    add_nonlinear_ls_mayer(
                        self,
                        J,
                        nlp.time_cx,
                        nlp.dt,
                        nlp.states.scaled.cx_start,
                        nlp.controls.scaled.cx_start,
                        nlp.parameters.scaled.cx,
                        nlp.algebraic_states.scaled.cx_start,
                        nlp.numerical_timeseries.cx,
                    )

            # Set costs
            self.acados_ocp.model.cost_y_expr = (
                self.lagrange_costs.reshape((-1, 1)) if self.lagrange_costs.numel() else SX(1, 1)
            )
            self.acados_ocp.model.cost_y_expr_e = (
                self.mayer_costs_e.reshape((-1, 1)) if self.mayer_costs_e.numel() else SX(1, 1)
            )
            self.acados_ocp.model.cost_y_expr_0 = (
                self.mayer_costs.reshape((-1, 1)) if self.mayer_costs.numel() else SX(1, 1)
            )

            # Set dimensions
            self.acados_ocp.dims.ny = self.acados_ocp.model.cost_y_expr.shape[0]
            self.acados_ocp.dims.ny_e = self.acados_ocp.model.cost_y_expr_e.shape[0]
            self.acados_ocp.dims.ny_0 = self.acados_ocp.model.cost_y_expr_0.shape[0]

            # Set weight
            self.acados_ocp.cost.W = np.zeros((1, 1)) if self.W.shape == (0, 0) else self.W
            self.acados_ocp.cost.W_e = np.zeros((1, 1)) if self.W_e.shape == (0, 0) else self.W_e
            self.acados_ocp.cost.W_0 = np.zeros((1, 1)) if self.W_0.shape == (0, 0) else self.W_0

            # Set target shape
            self.acados_ocp.cost.yref = np.zeros((self.acados_ocp.cost.W.shape[0],))
            self.acados_ocp.cost.yref_e = np.zeros((self.acados_ocp.cost.W_e.shape[0],))
            self.acados_ocp.cost.yref_0 = np.zeros((self.acados_ocp.cost.W_0.shape[0],))

        elif self.acados_ocp.cost.cost_type == "EXTERNAL":
            raise RuntimeError("EXTERNAL is not interfaced yet, please use NONLINEAR_LS")

        else:
            raise RuntimeError("Available acados cost type: 'LINEAR_LS', 'NONLINEAR_LS' and 'EXTERNAL'.")

    def __update_runtime_parameters(self) -> None:
        """Update node-wise numerical data without regenerating the Acados solver."""

        runtime_parameters = _get_acados_runtime_parameters(self.ocp.nlp[0])
        expected_size = int(self.acados_model.p.shape[0])
        if runtime_parameters.shape[0] != expected_size:
            raise RuntimeError(
                "The numerical time-series structure changed after the Acados model was generated. "
                f"Expected {expected_size} runtime parameters per node, got {runtime_parameters.shape[0]}. "
                "Create a new OptimalControlProgram to change their number."
            )

        if expected_size == 0:
            self._runtime_parameter_values = runtime_parameters
            return

        for stage in range(self.acados_ocp.solver_options.N_horizon + 1):
            stage_values = np.ascontiguousarray(runtime_parameters[:, stage], dtype=np.float64)
            if self._runtime_parameter_values is None:
                self.ocp_solver.set(stage, "p", stage_values)
                continue

            changed_indices = np.flatnonzero(stage_values != self._runtime_parameter_values[:, stage])
            if changed_indices.size == 0:
                continue
            if changed_indices.size == expected_size:
                self.ocp_solver.set(stage, "p", stage_values)
            else:
                self.ocp_solver.set_params_sparse(
                    stage,
                    changed_indices,
                    np.ascontiguousarray(stage_values[changed_indices], dtype=np.float64),
                )

        self._runtime_parameter_values = runtime_parameters.copy()

    def __update_solver(self):
        """
        Update the ACADOS solver to new values
        """

        self.__update_runtime_parameters()

        param_init = []
        for key in self.ocp.nlp[0].parameters.keys():
            scale_init = self.ocp.parameter_init[key].scale(self.ocp.parameters[key].scaling.scaling)
            param_init = np.concatenate((param_init, scale_init.init[:, 0]))

        for n in range(self.acados_ocp.solver_options.N_horizon):
            if n == 0:
                # Initial node
                if self.y_ref_start:
                    if len(self.y_ref_start) == 1:
                        self.ocp_solver.cost_set(0, "yref", np.array(self.y_ref_start[0])[:, 0])
                    else:
                        self.ocp_solver.cost_set(0, "yref", np.concatenate(self.y_ref_start)[:, 0])
                self.ocp_solver.constraints_set(0, "lbx", self.x_bound_min[:, 0])
                self.ocp_solver.constraints_set(0, "ubx", self.x_bound_max[:, 0])
            else:
                if self.y_ref:  # Target
                    self.ocp_solver.cost_set(n, "yref", np.vstack([data[n] for data in self.y_ref])[:, 0])
                self.ocp_solver.constraints_set(n, "lbx", self.x_bound_min[:, 1])
                self.ocp_solver.constraints_set(n, "ubx", self.x_bound_max[:, 1])

            # Intermediates
            # check following line
            # self.ocp_solver.cost_set(n, "W", self.W)

            # The x_init need to be ordered by index that's why we use a for loop
            x_init = np.ndarray((self.ocp.nlp[0].states.shape))
            for key in self.ocp.nlp[0].states.keys():
                index = self.ocp.nlp[0].states[key].index
                self.ocp.nlp[0].x_init[key].check_and_adjust_dimensions(
                    self.ocp.nlp[0].states[key].shape, self.ocp.nlp[0].ns
                )
                x_init[index] = (
                    self.ocp.nlp[0].x_init[key].init.evaluate_at(n) / self.ocp.nlp[0].x_scaling[key].scaling[:, 0]
                )

            self.ocp_solver.set(n, "x", np.concatenate((param_init, x_init)))

            # The u_init need to be ordered by index that's why we use a for loop
            u_init = np.ndarray((self.acados_ocp.dims.nu, 1))
            for key in self.ocp.nlp[0].controls.keys():
                index = self.ocp.nlp[0].controls[key].index
                self.ocp.nlp[0].u_init[key].check_and_adjust_dimensions(
                    self.ocp.nlp[0].controls[key].shape, self.ocp.nlp[0].ns - 1
                )
                u_init[index, 0] = (
                    self.ocp.nlp[0].u_init[key].init.evaluate_at(n) / self.ocp.nlp[0].u_scaling[key].scaling[:, 0]
                )

            self.ocp_solver.set(n, "u", u_init)

            # The u_bounds need to be ordered by index that's why we use a for loop
            u_bounds_max = np.ndarray(self.acados_ocp.dims.nu)
            u_bounds_min = np.ndarray(self.acados_ocp.dims.nu)
            for key in self.ocp.nlp[0].controls.keys():
                u_tp = self.ocp.nlp[0].u_bounds[key]
                index = self.ocp.nlp[0].controls[key].index
                u_bounds_max[index] = np.array(u_tp.max[:, 0])
                u_bounds_min[index] = np.array(u_tp.min[:, 0])

            self.ocp_solver.constraints_set(n, "lbu", u_bounds_min)
            self.ocp_solver.constraints_set(n, "ubu", u_bounds_max)
            self.ocp_solver.constraints_set(n, "uh", self.all_g_bounds.max[:, 0])
            self.ocp_solver.constraints_set(n, "lh", self.all_g_bounds.min[:, 0])

        # Final
        if self.y_ref_end:
            if len(self.y_ref_end) == 1:
                self.ocp_solver.cost_set(
                    self.acados_ocp.solver_options.N_horizon, "yref", np.array(self.y_ref_end[0])[:, 0]
                )
            else:
                self.ocp_solver.cost_set(
                    self.acados_ocp.solver_options.N_horizon, "yref", np.concatenate(self.y_ref_end)[:, 0]
                )
            # check following line
            # self.ocp_solver.cost_set(self.acados_ocp.solver_options.N_horizon, "W", self.W_e)
        self.ocp_solver.constraints_set(self.acados_ocp.solver_options.N_horizon, "lbx", self.x_bound_min[:, -1])
        self.ocp_solver.constraints_set(self.acados_ocp.solver_options.N_horizon, "ubx", self.x_bound_max[:, -1])

        if len(self.end_g_bounds.max[:, 0]):
            self.ocp_solver.constraints_set(self.acados_ocp.solver_options.N_horizon, "uh", self.end_g_bounds.max[:, 0])
            self.ocp_solver.constraints_set(self.acados_ocp.solver_options.N_horizon, "lh", self.end_g_bounds.min[:, 0])

        # The x_init need to be ordered by index that's why we use a for loop
        x_init = np.ndarray((self.ocp.nlp[0].states.shape,))
        for key in self.ocp.nlp[0].states.keys():
            index = self.ocp.nlp[0].states[key].index
            x_init[index] = self.ocp.nlp[0].x_init[key].init.evaluate_at(self.acados_ocp.solver_options.N_horizon)

        self.ocp_solver.set(self.acados_ocp.solver_options.N_horizon, "x", np.concatenate((param_init, x_init)))

    def online_optim(self, ocp):
        raise NotImplementedError("online_optim is not implemented yet with ACADOS backend")

    def get_optimized_value(self) -> AnyListorDict:
        """
        Get the previously optimized solution

        Returns
        -------
        A solution or a list of solution depending on the number of phases
        """

        ns = self.acados_ocp.solver_options.N_horizon
        n_params = self.ocp.nlp[0].parameters.shape
        acados_x = np.array([self.ocp_solver.get(i, "x") for i in range(ns + 1)]).T
        acados_p = acados_x[:n_params, :]
        acados_x = acados_x[n_params:, :]
        acados_u = np.array([self.ocp_solver.get(i, "u") for i in range(ns)]).T

        diagnostics = self.get_diagnostics()
        out = {
            "x": [],
            "u": acados_u,
            "solver_time_to_optimize": self.ocp_solver.get_stats("time_tot"),
            "real_time_to_optimize": self.real_time_to_optimize,
            "iter": self.ocp_solver.get_stats("sqp_iter"),
            "status": self.status,
            "solver": SolverType.ACADOS,
            "solver_diagnostics": diagnostics,
            "solver_state": self.get_solver_state(),
        }

        out["x"] = vertcat(out["x"], acados_x.reshape(-1, 1, order="F"))
        out["x"] = vertcat(out["x"], acados_u.reshape(-1, 1, order="F"))
        out["x"] = vertcat(out["x"], acados_p[:, 0])

        # Add dt to solution
        dt_init = self.ocp.dt_parameter_initial_guess.init[0, 0]
        dt = Function("dt", [self.ocp.nlp[0].dt], [self.ocp.nlp[0].dt])(dt_init)
        out["x"] = vertcat(dt, out["x"])

        self.out["sol"] = out
        out = []
        for key in self.out.keys():
            out.append(self.out[key])

        return out[0] if len(out) == 1 else out

    def get_solver_state(self) -> dict | None:
        """
        Capture a detached Acados primal-dual iterate for an exact warm start.

        The payload is intentionally opaque to the generic Solution API. Algebraic
        variables are not included until the Acados interface supports them.
        """

        if self.ocp_solver is None:
            return None

        iterate = self.ocp_solver.get_flat_iterate()
        return {
            "solver": SolverType.ACADOS.value,
            "format_version": _ACADOS_SOLVER_STATE_FORMAT_VERSION,
            "n_horizon": self.acados_ocp.solver_options.N_horizon,
            "iterate": {
                field: np.asarray(getattr(iterate, field), dtype=float).copy() for field in _ACADOS_WARM_START_FIELDS
            },
        }

    def set_lagrange_multiplier(self, sol: Solution) -> None:
        """
        Queue the Acados solver state carried by a Solution for the next solve.

        This method implements the common Bioptim warm-start hook. When the
        Solution has no Acados state, the existing state/control initial-guess
        path remains active and no dual iterate is restored.
        """

        solver_state = sol.solver_state
        if solver_state is None:
            self._warm_start_solver_state = None
            return

        if not isinstance(solver_state, dict) or solver_state.get("solver") != SolverType.ACADOS.value:
            raise ValueError("The warm-start solver state is not an Acados state.")
        if solver_state.get("format_version") != _ACADOS_SOLVER_STATE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported Acados warm-start state format "
                f"{solver_state.get('format_version')}; expected {_ACADOS_SOLVER_STATE_FORMAT_VERSION}."
            )
        if solver_state.get("n_horizon") != self.acados_ocp.solver_options.N_horizon:
            raise ValueError(
                "The Acados warm-start state uses a different number of shooting intervals. "
                "Grid-changing warm starts require interpolation and are not implemented yet."
            )

        iterate = solver_state.get("iterate")
        if not isinstance(iterate, dict) or any(field not in iterate for field in _ACADOS_WARM_START_FIELDS):
            raise ValueError("The Acados warm-start state must contain x, u, pi, lam, sl, and su flattened iterates.")
        self._warm_start_solver_state = deepcopy(solver_state)

    def __restore_solver_state(self) -> None:
        """Restore the queued Acados iterate after numerical inputs and primal guesses are updated."""

        if self._warm_start_solver_state is None:
            return

        solver_state = self._warm_start_solver_state
        self._warm_start_solver_state = None
        for field in _ACADOS_WARM_START_FIELDS:
            values = np.ascontiguousarray(solver_state["iterate"][field], dtype=float)
            expected_size = self.ocp_solver.get_dim_flat(field)
            if values.ndim != 1 or values.shape[0] != expected_size:
                raise ValueError(
                    f"The Acados warm-start field '{field}' has size {values.size}; expected {expected_size}. "
                    "The solver structure must match exactly for a primal-dual warm start."
                )
            if expected_size:
                self.ocp_solver.set_flat(field, values)

    def get_diagnostics(self) -> dict:
        """
        Get a snapshot of the public diagnostics from the previous Acados solve.

        The returned values are detached from the mutable Acados capsule, so they
        remain valid when the same solver is reused for a subsequent solve.
        """

        if self.ocp_solver is None:
            return {
                "status": self.status,
                "status_label": status_to_str(-1),
                "successful": False,
                "nlp_solver_type": self.opts.nlp_solver_type,
                "qp_solver": self.opts.qp_solver,
                "residuals": None,
                "nlp_iterations": None,
                "sqp_iterations": None,
                "qp_scaling_status": None,
                "timings": {},
                "raw_statistics": None,
                "qp_status": None,
                "qp_iterations": None,
                "qp_iterations_per_solve": None,
                "step_sizes": None,
            }

        return _collect_acados_diagnostics(
            self.ocp_solver,
            status=self.status,
            nlp_solver_type=self.acados_ocp.solver_options.nlp_solver_type,
            qp_solver=self.acados_ocp.solver_options.qp_solver,
        )

    def solve(self, expand_during_shake_tree: Bool = False) -> AnyListorDict:
        """
        Solve the prepared ocp

        Parameters
        ----------
        expand_during_shake_tree: bool
            If the casadi graph should be expanded during the shake tree phase. This value is ignored for ACADOS

        Returns
        -------
        A reference to the solution
        """

        tic = perf_counter()
        # Populate costs and constraints vectors
        self.__set_costs(self.ocp)
        self.__set_constraints(self.ocp)

        options = self.opts.as_dict(self)
        if self.ocp_solver is None:
            for key in options:
                setattr(self.acados_ocp.solver_options, key, options[key])
            self.ocp_solver = AcadosOcpSolver(
                self.acados_ocp,
                json_file=self.acados_ocp.code_gen_options.json_file,
                build=False if self.opts.check_reuse_possible else self.opts.c_compile,
                generate=not self.opts.check_reuse_possible,
                check_reuse_possible=self.opts.check_reuse_possible,
                tol_code_reuse=self.opts.tol_code_reuse,
            )
            self.opts.set_only_first_options_has_changed(False)
            self.opts.set_has_tolerance_changed(False)

        else:
            if self.opts.only_first_options_has_changed:
                raise RuntimeError(
                    "Some options has been changed the second time acados was run.",
                    "Only " + str(Solver.ACADOS.get_tolerance_keys()) + " can be modified.",
                )

            if self.opts.reset_solver_before_solve:
                self.ocp_solver.reset()

            if self.opts.has_tolerance_changed:
                for key in self.opts.get_tolerance_keys():
                    short_key = key[12:]
                    self.ocp_solver.options_set(short_key, options[key[1:]])
                self.opts.set_has_tolerance_changed(False)

        self.__update_solver()
        self.__restore_solver_state()
        self.status = self.ocp_solver.solve()
        self.real_time_to_optimize = perf_counter() - tic
        return self.get_optimized_value()
