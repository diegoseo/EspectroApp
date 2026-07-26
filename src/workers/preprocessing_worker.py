import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
import traceback
from PySide6.QtCore import QThread, Signal
from core.translations import get_language, translate_worker_error
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


class PreprocessingThread(QThread):
    """
    Applies a configurable sequence of preprocessing operations to spectral data in a background thread.
    The thread always executes baseline correction, normalization, smoothing, and derivative steps in a fixed order and emits a DataFrame in the app's internal format when finished.

    Parameters
    ----------
    df_original : pandas.DataFrame
        Original spectral DataFrame in the internal format, including the X-axis column and type/label row.
    options : dict
        Dictionary of preprocessing options and parameters indicating which operations to apply.
    """

    dataframe_result = Signal(object)
    error_signal = Signal(str)

    def __init__(self, df_original, options):
        super().__init__()
        self.df = df_original.copy()
        self.options = options

    def run(self):
        try:
            self._run_preprocessing()
        except Exception as error:
            self.error_signal.emit(
                translate_worker_error(error, get_language())
            )

    def _run_preprocessing(self):
        """
        Executes the configured preprocessing pipeline on the spectral data in a worker thread.
        The method applies each selected transformation, rebuilds the internal-format DataFrame, and emits the result when processing is complete.

        Returns
        -------
        None
        """
        df = self.df
        cabecera = df.iloc[0]
        df_without_header = df[1:].copy()
        raman_shift = df_without_header.iloc[:, 0].astype(float)
        intensity_df = df_without_header.iloc[:, 1:].apply(
            pd.to_numeric, errors="coerce"
        )

        linear_options = self.options.get("correccion_lineal")
        if linear_options:
            df = correct_linear_baseline(
                intensity_df,
                raman_shift,
                x_start=linear_options["x_start"],
                x_end=linear_options["x_end"],
            )
            intensity_df = df
        shirley_options = self.options.get("correccion_shirley")
        if shirley_options:
            df = correct_shirley_baseline(
                intensity_df,
                raman_shift,
                x_start=shirley_options["x_start"],
                x_end=shirley_options["x_end"],
                tolerance=shirley_options.get("tolerance", 1e-6),
                max_iterations=shirley_options.get(
                    "max_iterations",
                    100,
                ),
            )
            intensity_df = df
        if self.options.get("normalizar_media", {}).get("activar"):
            metodo = self.options["normalizar_media"]["metodo"]
            df = normalize_by_mean(intensity_df, metodo)
            intensity_df = df
        if self.options.get("normalizar_area"):
            df = normalize_by_area(intensity_df, raman_shift)
            intensity_df = df
        if self.options.get("suavizar_sg"):
            ventana = self.options["suavizar_sg"]["ventana"]
            orden = self.options["suavizar_sg"]["orden"]
            df = smooth_savitzky_golay(intensity_df, ventana, orden)
            intensity_df = df
        if self.options.get("suavizar_fg"):
            sigma = self.options["suavizar_fg"]["sigma"]
            df = smooth_gaussian_filter(intensity_df, sigma)
            intensity_df = df
        if self.options.get("suavizar_mm"):
            ventana = self.options["suavizar_mm"]["ventana"]
            df = smooth_moving_average(intensity_df, ventana)
            intensity_df = df
        if self.options.get("derivada_1"):
            df = calculate_first_derivative(intensity_df, raman_shift)
            intensity_df = df
        if self.options.get("derivada_2"):
            df = calculate_second_derivative(intensity_df, raman_shift)
            intensity_df = df

        nombre_eje_x = str(cabecera.iloc[0]).strip()

        df_final = pd.concat(
            [
                raman_shift.reset_index(drop=True),
                intensity_df.reset_index(drop=True),
            ],
            axis=1,
        )

        df_final.columns = [nombre_eje_x] + list(cabecera.iloc[1:])
        df_final = pd.concat(
            [pd.DataFrame([df_final.columns], columns=df_final.columns), df_final],
            ignore_index=True,
        )

        n_cols = df_final.shape[1]
        df_final.columns = [0] + list(range(1, n_cols))
        df_final.attrs = self.df.attrs.copy()
        history = list(df_final.attrs.get("preprocessing_history", []))
        history.append(self.options.copy())
        df_final.attrs["preprocessing_history"] = history
        self.dataframe_result.emit(df_final)