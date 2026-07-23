import json
import multiprocessing as mp
import threading
from queue import Empty

from casadi import nlpsol_out, DM
from matplotlib import pyplot as plt
import numpy as np

from .plot import PlotOcp, OcpSerializable
from .online_callback_server import _deserialize_xydata, _serialize_xydata
from ..optimization.optimization_vector import OptimizationVectorHelper
from .online_callback_abstract import OnlineCallbackAbstract
from ..misc.parameters_types import (
    Bool,
    Float,
    AnyIterable,
    AnyDictOptional,
    IntListOptional,
)


class OnlineCallbackMultiprocess(OnlineCallbackAbstract):
    """
    Multiprocessing implementation of the online callback

    Attributes
    ----------
    queue: mp.Queue
        The multiprocessing queue
    plotter: ProcessPlotter
        The callback for plotting for the multiprocessing
    plot_process: mp.Process
        The multiprocessing placeholder
    """

    def __init__(self, ocp, opts: AnyDictOptional = None, **show_options):
        super(OnlineCallbackMultiprocess, self).__init__(ocp, opts, **show_options)

        if self.ocp.plot_ipopt_outputs:
            raise NotImplementedError(
                "The multiprocess online callback does not support the plot_ipopt_outputs option when using a "
                "serialized plotting process"
            )
        if self.ocp.plot_check_conditioning:
            raise NotImplementedError(
                "The multiprocess online callback does not support the plot_check_conditioning option when using a "
                "serialized plotting process"
            )

        dummy_phase_times = OptimizationVectorHelper.extract_step_times(
            self.ocp, DM(np.ones(self.ocp.n_phases))
        )
        self._plotter = PlotOcp(
            self.ocp,
            only_initialize_variables=True,
            dummy_phase_times=dummy_phase_times,
            **show_options,
        )

        ocp_for_plotting = OcpSerializable.from_ocp(self.ocp)
        ocp_for_plotting.save_ipopt_iterations_info = None
        serialized_ocp = json.dumps(ocp_for_plotting.serialize()).encode()
        serialized_dummy_phase_times = [
            [np.array(node_times)[:, 0].tolist() for node_times in phase_times]
            for phase_times in dummy_phase_times
        ]

        self.queue = mp.Queue()
        self.plotter = self.ProcessPlotter(serialized_ocp, serialized_dummy_phase_times)
        self.plot_process = mp.Process(
            target=self.plotter, args=(self.queue, show_options), daemon=True
        )
        self.plot_process.start()

    def close(self) -> None:
        self.plot_process.kill()

    def eval(self, arg: AnyIterable, enforce: Bool = False) -> IntListOptional:
        # Dequeuing the data by removing previous not useful data
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break

        args_dict = {}
        for i, s in enumerate(nlpsol_out()):
            args_dict[s] = arg[i]

        if self.ocp.save_ipopt_iterations_info is not None:
            from ..gui.ipopt_output_plot import save_ipopt_output

            save_ipopt_output(args_dict, self.ocp.save_ipopt_iterations_info)

        xdata, ydata = self._plotter.parse_data(**args_dict)
        self.queue.put_nowait(_serialize_xydata(xdata, ydata))
        return [0]

    class ProcessPlotter(object):
        """
        The plotter that interface PlotOcp and the multiprocessing

        Attributes
        ----------
        _serialized_ocp: bytes
            The serialized description of the plots
        _dummy_phase_times: list
            The numerical time vector used to initialize the plots
        _plotter: PlotOcp
            The plotter
        _update_time: float
            The time between each update

        Methods
        -------
        callback(self) -> bool
            The callback to update the graphs
        """

        def __init__(self, serialized_ocp: bytes, dummy_phase_times: list):
            """
            Parameters
            ----------
            serialized_ocp: bytes
                The serialized description of the plots
            dummy_phase_times: list
                The numerical time vector used to initialize the plots
            """

            self._serialized_ocp = serialized_ocp
            self._dummy_phase_times = dummy_phase_times
            self._plotter: PlotOcp = None
            self._update_time: Float = 0.001

        def __call__(self, pipe: mp.Queue, show_options: AnyDictOptional):
            """
            Parameters
            ----------
            pipe: mp.Queue
                The multiprocessing queue to evaluate
            show_options: dict
                The option to pass to PlotOcp
            """

            if show_options is None:
                show_options = {}
            self._pipe = pipe

            self._initialize_plotter(show_options)
            threading.Timer(self._update_time, self.plot_update).start()
            plt.show()

        def _initialize_plotter(self, show_options: dict) -> None:
            """Reconstruct the plotting-only OCP and initialize its figures."""

            ocp = OcpSerializable.deserialize(json.loads(self._serialized_ocp))
            dummy_phase_times = [
                [DM(node_times) for node_times in phase_times]
                for phase_times in self._dummy_phase_times
            ]
            self._plotter = PlotOcp(
                ocp, dummy_phase_times=dummy_phase_times, **show_options
            )

        @property
        def has_at_least_one_active_figure(self) -> Bool:
            """
            If at least one figure is active

            Returns
            -------
            If at least one figure is active
            """

            return [
                plt.fignum_exists(fig.number) for fig in self._plotter.all_figures
            ].count(True) > 0

        def plot_update(self) -> Bool:
            """
            The callback to update the graphs

            Returns
            -------
            True if everything went well
            """

            serialized_xydata = None
            while True:
                try:
                    serialized_xydata = self._pipe.get_nowait()
                except Empty:
                    break

            if serialized_xydata is not None:
                self._plotter.update_data(*_deserialize_xydata(serialized_xydata))

            # We want to redraw here to actually consume a bit of time, otherwise it goes to fast and pipe remains empty
            for fig in self._plotter.all_figures:
                fig.canvas.draw()
            if self.has_at_least_one_active_figure:
                # If there are still figures, we keep updating
                threading.Timer(self._update_time, self.plot_update).start()

            return True
