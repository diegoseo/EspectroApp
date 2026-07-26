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


class DimensionalityReductionThread(QThread):
    """
    Runs PCA, t-SNE, and related dimensionality reduction workflows in a background thread.
    The thread generates figures for selected projections, loading plots, and an optional report, emitting each result via Qt signals for display in the GUI.

    Parameters
    ----------
    df_original : pandas.DataFrame
        Original spectral DataFrame in the internal format, including the X-axis column and type/label row.
    options : dict
        Dictionary of user-selected options indicating which dimensionality reduction methods and plots to compute.
    componentes : int or str
        Number of target components for PCA or t-SNE, often coming from a GUI input field.
    intervalo : int or float
        Confidence interval or opacity scaling factor used when plotting score clouds.
    nombre_informe : str
        Base name or path for the report file to generate when the corresponding option is enabled.
    componentes_seleccionados : list of int
        Indices of principal components or dimensions to visualize in 2D/3D scatter plots or heatmaps.
    cp_pca : int
        Number of PCA components to retain before applying t-SNE in the t-SNE(PCA(X)) workflow.
    cp_tsne : int
        Number of t-SNE output dimensions (typically 2 or 3) for the t-SNE(PCA(X)) workflow.
    componentes_selec_loading : list of int
        Indices of principal components to include in the loading plot.
    cant_componentes_loading : int or str
        Number of loading components to highlight or summarize in the loading figure.
    """

    pca_2d_figure_signal = Signal(object)
    pca_3d_figure_signal = Signal(object)

    tsne_2d_figure_signal = Signal(object)
    tsne_3d_figure_signal = Signal(object)

    tsne_pca_2d_figure_signal = Signal(object)
    tsne_pca_3d_figure_signal = Signal(object)

    loading_figure_signal = Signal(object)
    pca_model_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(
        self,
        df_original,
        options,
        componentes,
        intervalo,
        nombre_informe,
        componentes_seleccionados,
        cp_pca,
        cp_tsne,
        componentes_selec_loading,
        cant_componentes_loading,
        tsne_parameters=None,
    ):
        super().__init__()
        self.df = df_original.copy()
        self.df.attrs = df_original.attrs.copy()
        self.options = options
        self.componentes = componentes
        self.intervalo = intervalo
        self.nombre_informe = nombre_informe
        self.cp_pca = cp_pca
        self.cp_tsne = cp_tsne
        self.componentes_seleccionados = componentes_seleccionados
        self.componentes_selec_loading = componentes_selec_loading
        self.cant_componentes_loading = cant_componentes_loading
        self.tsne_parameters = tsne_parameters or {}
        self.tipos = self.df.iloc[0, 1:]

        self.color_mapping = assign_type_colors(self.tipos)

        self.raman_shift = self.df.iloc[1:, 0].reset_index(drop=True)
        self.explained_variance_percentage = None
        self.pca_resultado = None
        self.tsne_resultado = None
        self.tsne_pca_resultado = None

    def run(self):
        try:
            self.run_analysis()

        except Exception as error:
            print("\n[T-SNE/PCA ERROR]")
            traceback.print_exc()

            self.error_signal.emit(
                translate_worker_error(
                    f"{type(error).__name__}: {error}",
                    get_language(),
                )
            )

    def run_analysis(self):

        if self.options.get("PCA"):
            print(
                "[PCA] DataFrame recibido:",
                self.df.shape,
            )

            print(
                "[PCA] Primera fila:",
                self.df.iloc[0, :5].tolist(),
            )

            X = prepare_pca_matrix(self.df)

            print(
                "[PCA] Matriz numérica preparada:",
                X.shape,
                X.dtype,
            )

            self.pca_resultado, self.explained_variance_percentage, pca_model = pca(
                X,
                self.componentes,
                return_model=True,
            )
            self.pca_model_signal.emit({
                "model": pca_model,
                "n_features": int(X.shape[1]),
                "n_components": int(self.componentes),
                "feature_axis": self.raman_shift.to_numpy(copy=True),
            })

            print(
                "[PCA] Cálculo terminado:",
                self.pca_resultado.shape,
            )

            if self.options.get("GRAFICO 2D"):
                pc_x, pc_y = self.componentes_seleccionados["2d"]
                fig = plot_pca_2d(
                    self.pca_resultado,
                    self.explained_variance_percentage,
                    self.color_mapping,
                    self.tipos,
                    pc_x,
                    pc_y,
                    self.intervalo,
                )
                self.pca_2d_figure_signal.emit(fig)

            if self.options.get("GRAFICO 3D"):
                pc_x, pc_y, pc_z = self.componentes_seleccionados["3d"]
                fig = plot_pca_3d(
                    self.pca_resultado,
                    self.explained_variance_percentage,
                    self.color_mapping,
                    self.tipos,
                    pc_x,
                    pc_y,
                    pc_z,
                    self.intervalo,
                )
                self.pca_3d_figure_signal.emit(fig)
        if self.options.get("TSNE"):
            print("[T-SNE] DataFrame recibido:", self.df.shape)

            df_intensidades = (
                self.df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").T
            )

            print("[T-SNE] Matriz numérica:", df_intensidades.shape)

            if df_intensidades.isna().any().any():
                raise ValueError(
                    "The selected dataset contains "
                    "non-numeric or missing intensity values."
                )

            direct_dimensions = int(self.tsne_parameters.get("direct_dimensions", 2))

            direct_perplexity = float(self.tsne_parameters.get("direct_perplexity", 30))

            direct_iterations = int(self.tsne_parameters.get("direct_iterations", 1000))

            print(
                "[T-SNE] Parámetros:",
                "dimensiones =",
                direct_dimensions,
                "perplexity =",
                direct_perplexity,
                "iteraciones =",
                direct_iterations,
            )

            self.tsne_resultado = tsne(
                df_intensidades,
                n_componentes=direct_dimensions,
                perplexity=direct_perplexity,
                max_iter=direct_iterations,
            )

            print(
                "[T-SNE] Resultado calculado:",
                self.tsne_resultado.shape,
            )

            if self.options.get("GRAFICO 2D") and direct_dimensions >= 2:
                fig = plot_tsne_2d(
                    self.tsne_resultado[:, :2],
                    self.tipos,
                    self.color_mapping,
                    float(self.intervalo) / 100,
                )

                self.tsne_2d_figure_signal.emit(fig)
                print("[T-SNE] Figura 2D enviada.")

            if self.options.get("GRAFICO 3D") and direct_dimensions >= 3:
                fig = plot_tsne_3d(
                    self.tsne_resultado[:, :3],
                    self.tipos,
                    self.color_mapping,
                    float(self.intervalo) / 100,
                )

                self.tsne_3d_figure_signal.emit(fig)
                print("[T-SNE] Figura 3D enviada.")

        if self.options.get("t-SNE(PCA(X))"):
            print(
                "[t-SNE(PCA)] DataFrame recibido:",
                self.df.shape,
            )

            df_intensidades = (
                self.df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").T
            )

            print(
                "[t-SNE(PCA)] Matriz numérica:",
                df_intensidades.shape,
            )

            if df_intensidades.isna().any().any():
                raise ValueError(
                    "The selected dataset contains "
                    "non-numeric or missing intensity values."
                )

            pca_perplexity = float(self.tsne_parameters.get("pca_perplexity", 30))

            pca_iterations = int(self.tsne_parameters.get("pca_iterations", 1000))

            print(
                "[t-SNE(PCA)] Parámetros:",
                "PCs PCA =",
                self.cp_pca,
                "dimensiones t-SNE =",
                self.cp_tsne,
                "perplexity =",
                pca_perplexity,
                "iteraciones =",
                pca_iterations,
            )

            self.tsne_pca_resultado = tsne_pca(
                df_intensidades,
                self.cp_pca,
                self.cp_tsne,
                perplexity=pca_perplexity,
                max_iter=pca_iterations,
            )

            print(
                "[t-SNE(PCA)] Resultado calculado:",
                self.tsne_pca_resultado.shape,
            )

            if self.options.get("GRAFICO 2D") and self.cp_tsne >= 2:
                fig = plot_tsne_2d(
                    self.tsne_pca_resultado[:, :2],
                    self.tipos,
                    self.color_mapping,
                    float(self.intervalo) / 100,
                )

                self.tsne_pca_2d_figure_signal.emit(fig)
                print("[t-SNE(PCA)] Figura 2D enviada.")

            if self.options.get("GRAFICO 3D") and self.cp_tsne >= 3:
                fig = plot_tsne_3d(
                    self.tsne_pca_resultado[:, :3],
                    self.tipos,
                    self.color_mapping,
                    float(self.intervalo) / 100,
                )

                self.tsne_pca_3d_figure_signal.emit(fig)
                print("[t-SNE(PCA)] Figura 3D enviada.")

        if self.options.get("Grafico Loading (PCA)"):

            if self.df.attrs.get("fusion_type") == "low_level":
                fig = plot_low_level_fusion_loadings(
                    self.df, self.componentes_selec_loading
                )

            else:
                x = pd.to_numeric(
                    self.df.iloc[1:, 0].reset_index(drop=True), errors="coerce"
                )

                Y = (
                    self.df.iloc[1:, 1:]
                    .apply(pd.to_numeric, errors="coerce")
                    .reset_index(drop=True)
                )

                valid_x = x.notna()
                x = x[valid_x].reset_index(drop=True)
                Y = Y.loc[valid_x].reset_index(drop=True)

                valid_variables = ~Y.isna().any(axis=1)
                x = x[valid_variables].reset_index(drop=True)
                Y = Y.loc[valid_variables].reset_index(drop=True)

                valid_samples = ~Y.isna().any(axis=0)
                Y = Y.loc[:, valid_samples]

                X = Y.T.to_numpy(dtype=float)

                fig = plot_loadings(X, x, self.componentes_selec_loading)

            self.loading_figure_signal.emit(fig)
        if self.options.get("GENERAR INFORME"):
            generate_report(
                self.nombre_informe,
                self.options,
                self.componentes,
                self.intervalo,
                self.cp_pca,
                self.cp_tsne,
                self.componentes_seleccionados,
                self.color_mapping,
                self.pca_resultado,
                self.explained_variance_percentage,
                self.tsne_resultado,
                self.tsne_pca_resultado,
                tsne_parameters=self.tsne_parameters,
            )
