"""Background worker for hierarchical cluster analysis."""

from PySide6.QtCore import QThread, Signal

from functions import calculate_hca
from core.translations import get_language, translate_worker_error


class HcaThread(QThread):
    """Compute HCA without blocking the graphical interface."""

    # New signal used by the result page.
    signal_resultado_hca = Signal(object, object)
    # Compatibility signal retained for older connections.
    signal_figura_hca = Signal(object)
    error_signal = Signal(str)

    def __init__(self, df_original, options):
        super().__init__()
        self.df = df_original.copy()
        self.options = options
        self.raman_shift = self.df.iloc[1:, 0].reset_index(drop=True)
        self.muestras_hca = self.df.iloc[0, 1:].tolist()

    def run(self):
        try:
            fig, cluster_table = calculate_hca(
                self.df,
                self.raman_shift,
                self.options,
                self.muestras_hca,
            )
            self.signal_resultado_hca.emit(fig, cluster_table)
            self.signal_figura_hca.emit(fig)
        except Exception as error:
            self.error_signal.emit(translate_worker_error(error, get_language()))
