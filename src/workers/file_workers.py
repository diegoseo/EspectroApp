import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
import traceback
from PySide6.QtCore import QThread, Signal
from core.translations import get_language, translate
from file_handling import load_file, load_multiple_spa_if_x_matches
from functions import (
    correct_linear_baseline,
    correct_shirley_baseline,
    normalize_by_mean,
    normalize_by_area,
    smooth_savitzky_golay,
    smooth_gaussian_filter,
    smooth_moving_average,
    calculate_first_derivative,
    calculate_second_derivative,
    pca,
    plot_pca_2d,
    plot_pca_3d,
    tsne,
    plot_tsne_2d,
    plot_tsne_3d,
    tsne_pca,
    generate_report,
    calculate_hca,
    plot_loadings,
    plot_low_level_fusion_loadings,
    sort_samples,
    concatenate_low_level_fusion,
    concatenate_low_level_fusion_without_intersection,
    concatenate_mid_level_fusion,
    concatenate_mid_level_fusion_without_intersection,
    plot_heatmap_pca,
    prepare_pca_matrix,
    assign_type_colors,
)


class FileLoaderThread(QThread):
    """
    Loads one or more spectral files asynchronously in a background thread.
    The thread emits a signal with each loaded DataFrame, optionally merging multiple SPA files that share the same X-axis.

    Parameters
    ----------
    file_paths : list of str
        List of file paths to be loaded; may contain CSV and SPA files.
    parent : QObject, optional
        Parent Qt object that will own this thread instance, by default None.
    """

    file_loaded = Signal(object, str)

    def __init__(self, file_paths, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths

    def run(self):
        """
        Loads the configured files in the background and emits each result as a DataFrame.
        The method automatically merges multiple SPA files with a common X-axis or loads other supported formats individually, reporting any errors to the console.

        Returns
        -------
        None
        """
        try:
            extensiones = [
                os.path.splitext(ruta)[1].lower() for ruta in self.file_paths
            ]

            if len(self.file_paths) > 1 and all(ext == ".spa" for ext in extensiones):
                df = load_multiple_spa_if_x_matches(self.file_paths)
                self.file_loaded.emit(
                    df,
                    translate(
                        "SPA Fusion ({count} files)",
                        get_language(),
                        count=len(self.file_paths),
                    ),
                )
            else:
                for ruta in self.file_paths:
                    df = load_file(ruta)
                    self.file_loaded.emit(df, ruta)

        except Exception as e:
            print(translate("File loading error: {error}", get_language(), error=e))


class SpectraPlotThread(QThread):
    """
    Sends spectral data to the GUI thread for plotting without blocking the main application.
    The thread simply emits a signal carrying the spectra matrix, X-axis values, and color mapping so that a plotting widget can update asynchronously.

    Parameters
    ----------
    datos : pandas.DataFrame
        DataFrame in the internal format containing spectra to be plotted.
    raman_shift : array-like
        X-axis values (e.g., Raman shift) corresponding to the spectral measurements.
    color_mapping : dict
        Mapping from sample type or class name to color values used when rendering the spectra.
    """

    plot_signal = Signal(object, object, object)

    def __init__(self, datos, raman_shift, color_mapping):
        super().__init__()
        self.datos = datos
        self.raman_shift = raman_shift
        self.color_mapping = color_mapping

    def run(self):
        self.plot_signal.emit(self.datos, self.raman_shift, self.color_mapping)