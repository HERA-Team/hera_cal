# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("hera-calibration")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

del version
del PackageNotFoundError

from . import utils
from . import redcal
from . import io
from . import delay_filter
from . import abscal
from . import lstbin
from . import smooth_cal
from . import apply_cal
from . import frf
from . import flag_utils
from . import reflections
from . import vis_clean
from . import autos
from . import noise
from . import tempcal
