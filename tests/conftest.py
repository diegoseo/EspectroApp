import sys
import types
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def numeric_df():
    """DataFrame numérico reutilizable para las pruebas de preprocesamiento."""
    return pd.DataFrame(
        {
            "Sample_A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Sample_B": [2.0, 4.0, 6.0, 8.0, 10.0],
            "Sample_C": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )


try:
    import PySide6  # noqa: F401
except ModuleNotFoundError:
    pyside6 = types.ModuleType("PySide6")
    qtcore = types.ModuleType("PySide6.QtCore")

    class QObject:
        def __init__(self, parent=None):
            self.parent = parent

    class _BoundSignal:
        def connect(self, callback):
            self.callback = callback

        def emit(self, *args, **kwargs):
            callback = getattr(self, "callback", None)
            if callback:
                callback(*args, **kwargs)

    class Signal:
        def __init__(self, *args, **kwargs):
            self._signal = _BoundSignal()

        def __get__(self, instance, owner):
            return self._signal

    class QSettings:
        def __init__(self, *args, **kwargs):
            pass

        def value(self, key, default=None):
            return default

        def setValue(self, key, value):
            pass

    qtcore.QObject = QObject
    qtcore.Signal = Signal
    qtcore.QSettings = QSettings

    pyside6.QtCore = qtcore

    sys.modules["PySide6"] = pyside6
    sys.modules["PySide6.QtCore"] = qtcore


# dimensionality.py only needs calculate_accuracy from plotting during these
# algorithm tests. Avoid importing the full Qt plotting module when unavailable.
try:
    import pyqtgraph  # noqa: F401
except ModuleNotFoundError:
    plotting = types.ModuleType("plotting")

    def calculate_accuracy(dataframe, labels):
        return 100.0

    plotting.calculate_accuracy = calculate_accuracy
    sys.modules["plotting"] = plotting