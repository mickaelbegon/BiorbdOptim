import json
import multiprocessing as mp
import pickle
from multiprocessing.reduction import ForkingPickler

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from bioptim.gui.online_callback_server import _serialize_xydata, _deserialize_xydata
from bioptim.gui.online_callback_multiprocess import OnlineCallbackMultiprocess
from bioptim.gui.plot import PlotOcp
from bioptim.gui.online_callback_server import _ResponseHeader
from bioptim.gui.serializable_class import OcpSerializable
from bioptim.optimization.optimization_vector import OptimizationVectorHelper
from casadi import DM
from matplotlib import pyplot as plt

from ..utils import TestUtils


def _initialize_serialized_plotter(process_plotter, serialized_xydata, result_queue):
    """Exercise the complete child-side initialization in a fresh interpreter."""

    try:
        process_plotter._initialize_plotter({"automatically_organize": False})
        process_plotter._plotter.update_data(*_deserialize_xydata(serialized_xydata))
        result_queue.put(
            {
                "ocp_type": type(process_plotter._plotter.ocp).__name__,
                "n_figures": len(process_plotter._plotter.all_figures),
            }
        )
    except Exception as exc:
        result_queue.put({"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        plt.close("all")


def _prepare_ocp():
    from bioptim.examples.getting_started import basic_ocp as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    return ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=40,
    )


def test_serialize_deserialize():
    # Prepare a set of data to serialize and deserialize
    ocp = _prepare_ocp()

    dummy_phase_times = OptimizationVectorHelper.extract_step_times(
        ocp, DM(np.ones(ocp.n_phases))
    )
    plotter = PlotOcp(
        ocp,
        dummy_phase_times=dummy_phase_times,
        show_bounds=True,
        only_initialize_variables=True,
    )

    np.random.seed(42)
    xdata, ydata = plotter.parse_data(
        **{"x": np.random.rand(ocp.variables_vector.shape[0])[:, None]}
    )

    # Serialize and deserialize the data
    serialized_data = _serialize_xydata(xdata, ydata)
    deserialized_xdata, deserialized_ydata = _deserialize_xydata(serialized_data)

    # Compare the outputs
    for x_phase, deserialized_x_phase in zip(xdata, deserialized_xdata):
        for x_node, deserialized_x_node in zip(x_phase, deserialized_x_phase):
            assert np.allclose(x_node, DM(deserialized_x_node))

    for y_variable, deserialized_y_variable in zip(ydata, deserialized_ydata):
        if isinstance(y_variable, np.ndarray):
            assert np.allclose(y_variable, deserialized_y_variable[0], equal_nan=True)
        else:
            for y_phase, deserialized_y_phase in zip(
                y_variable, deserialized_y_variable
            ):
                assert np.allclose(y_phase, deserialized_y_phase)


def test_multiprocess_plotter_is_picklable_without_the_full_ocp():
    ocp = _prepare_ocp()
    dummy_phase_times = OptimizationVectorHelper.extract_step_times(
        ocp, DM(np.ones(ocp.n_phases))
    )
    serialized_dummy_phase_times = [
        [np.array(node_times)[:, 0].tolist() for node_times in phase_times]
        for phase_times in dummy_phase_times
    ]
    serialized_ocp = json.dumps(OcpSerializable.from_ocp(ocp).serialize()).encode()

    process_plotter = OnlineCallbackMultiprocess.ProcessPlotter(
        serialized_ocp, serialized_dummy_phase_times
    )
    process_plotter = pickle.loads(ForkingPickler.dumps(process_plotter))

    assert process_plotter._serialized_ocp == serialized_ocp
    assert not hasattr(process_plotter, "_ocp")

    process_plotter._initialize_plotter({"automatically_organize": False})
    assert isinstance(process_plotter._plotter.ocp, OcpSerializable)
    assert len(process_plotter._plotter.all_figures) > 0
    plt.close("all")


@pytest.mark.skipif(
    "forkserver" not in mp.get_all_start_methods(),
    reason="The forkserver start method is not available on this platform",
)
def test_multiprocess_plotter_runs_with_forkserver():
    ocp = _prepare_ocp()
    dummy_phase_times = OptimizationVectorHelper.extract_step_times(
        ocp, DM(np.ones(ocp.n_phases))
    )
    parent_plotter = PlotOcp(
        ocp,
        dummy_phase_times=dummy_phase_times,
        only_initialize_variables=True,
    )
    np.random.seed(42)
    xdata, ydata = parent_plotter.parse_data(
        **{"x": np.random.rand(ocp.variables_vector.shape[0])[:, None]}
    )

    serialized_dummy_phase_times = [
        [np.array(node_times)[:, 0].tolist() for node_times in phase_times]
        for phase_times in dummy_phase_times
    ]
    serialized_ocp = json.dumps(OcpSerializable.from_ocp(ocp).serialize()).encode()
    process_plotter = OnlineCallbackMultiprocess.ProcessPlotter(
        serialized_ocp, serialized_dummy_phase_times
    )

    context = mp.get_context("forkserver")
    result_queue = context.Queue()
    process = context.Process(
        target=_initialize_serialized_plotter,
        args=(process_plotter, _serialize_xydata(xdata, ydata), result_queue),
    )

    try:
        process.start()
        process.join(timeout=30)

        assert not process.is_alive(), "The forkserver plotting process did not stop"
        assert process.exitcode == 0
        result = result_queue.get(timeout=5)
        assert "error" not in result
        assert result["ocp_type"] == "OcpSerializable"
        assert result["n_figures"] > 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        result_queue.close()


def test_response_header():
    # Make sure all the response have the same length
    response_len = _ResponseHeader.response_len()
    for response in _ResponseHeader:
        assert len(response) == response_len
        # Make sure encoding provides a constant length
        assert len(response.encode()) == response_len

    # Make sure equality works
    assert _ResponseHeader.OK == _ResponseHeader.OK
    assert _ResponseHeader.OK.value == _ResponseHeader.OK
    assert _ResponseHeader.OK.encode().decode() == _ResponseHeader.OK
    assert _ResponseHeader.OK == _ResponseHeader.OK.encode().decode()
    assert not (_ResponseHeader.OK != _ResponseHeader.OK)
    assert not (_ResponseHeader.OK.encode().decode() != _ResponseHeader.OK)
    assert not (_ResponseHeader.OK != _ResponseHeader.OK.encode().decode())
    assert not (_ResponseHeader.OK.value == _ResponseHeader.OK.encode().decode())
    assert _ResponseHeader.OK != _ResponseHeader.NOK
    assert _ResponseHeader.NOK == _ResponseHeader.NOK
