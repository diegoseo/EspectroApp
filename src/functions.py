"""Backward-compatible public API for EspectroApp.

The implementation has been separated into focused modules under
``algorithms``. Existing imports such as
``from functions import normalize_by_mean`` continue to work unchanged.
"""

from algorithms.preprocessing import *
from algorithms.dimensionality import *
from algorithms.reporting import *
from algorithms.clustering import *
from algorithms.loadings import *
from algorithms.fusion import *
from algorithms.utilities import *
