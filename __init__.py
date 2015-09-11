import logging
import sys

logger = logging.getLogger(__name__)

try:
    import IPython
except ImportError:
    IPython = None

from .radialprofile import (find_ellipse, find_ellipsoid, refine_ellipse,
                            refine_ellipsoid, locate_ellipsoid,
                            batch_ellipsoid)
from .fluctuation import (powerspectrum, circle_deviation, fluctuation_spectrum,
                          correlation_time, powerspectrum_part,
                          efluctuation_spectrum, fit_fluctuation_kappa,
                          fit_fluctuation_sigma, fit_fluctuation_2param)
from .plot import (annotate_ellipsoid, annotate_ellipse,
                   drawEllipsoid, plot3D_vesicle_features,
                   plot_fluctuation_spectrum)
from .test import (gen_artificial_ellipsoid, gen_fluct_vesicle)
from .tracking import (Track, Features, track_multicore)
from .analysis import VesicleFeatures
from .utils import (export, gen_nd2_paths, image_md, image_loc, crop_pad,
                    play_file, exec_cluster)


class IPythonStreamHandler(logging.StreamHandler):
    "A StreamHandler for logging that clears output between entries."
    def emit(self, s):
        IPython.core.display.clear_output(wait=True)
        print(s.getMessage())
    def flush(self):
        sys.stdout.flush()

FORMAT = "%(name)s.%(funcName)s:  %(message)s"
formatter = logging.Formatter(FORMAT)

# Check for IPython and use a special logger
use_ipython_handler = False

if IPython is not None:
    if IPython.get_ipython() is not None:
        use_ipython_handler = True

if use_ipython_handler:
    default_handler = IPythonStreamHandler()
else:
    default_handler = logging.StreamHandler(sys.stdout)

default_handler.setLevel(logging.INFO)
default_handler.setFormatter(formatter)


def handle_logging():
    "Send INFO-level log messages to stdout. Do not propagate."
    if use_ipython_handler:
        # Avoid double-printing messages to IPython stderr.
        logger.propagate = False
    logger.addHandler(default_handler)
    logger.setLevel(logging.INFO)


def ignore_logging():
    "Reset to factory default logging configuration; remove trackpy's handler."
    logger.removeHandler(default_handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def quiet(suppress=True):
    """Suppress trackpy information log messages.

    Parameters
    ----------
    suppress : boolean
        If True, set the logging level to WARN, hiding INFO-level messages.
        If False, set level to INFO, showing informational messages.
    """
    if suppress:
        logger.setLevel(logging.WARN)
    else:
        logger.setLevel(logging.INFO)
        
handle_logging()
