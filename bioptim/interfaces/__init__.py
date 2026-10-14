from .ipopt_options import IPOPT
from .fratrop_options import FATROP
from .acados_options import ACADOS
from .sqp_options import SQP_METHOD
from .alpaqa_options import ALPAQA


class Solver:
    IPOPT = IPOPT
    FATROP = FATROP
    ALPAQA = ALPAQA
    SQP_METHOD = SQP_METHOD
    ACADOS = ACADOS
