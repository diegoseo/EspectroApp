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


class DataFusionThread(QThread):
    """
    Computes basic range and ordering information needed for data fusion of multiple spectral datasets.
    The thread analyzes the selected DataFrames, determines common and individual ranges, and emits the results for use in fusion workflows.

    Parameters
    ----------
    df_seleccionados : list of pandas.DataFrame
        List of spectral DataFrames selected by the user for data fusion operations.
    """

    signal_datafusion = Signal(object, object, object, object)

    def __init__(self, df_seleccionados):
        super().__init__()
        self.df_seleccionados = df_seleccionados

    def run(self):

        lista_rangos, interseccion, rang_comun, tipos_orden = sort_samples(
            self.df_seleccionados
        )

        self.signal_datafusion.emit(lista_rangos, interseccion, rang_comun, tipos_orden)


class LowLevelDataFusionThread(QThread):
    """
    Performs low-level data fusion of multiple spectral datasets in a background thread.
    The thread concatenates spectra according to user-selected ranges and interpolation settings, then emits the fused DataFrame for further analysis.

    Parameters
    ----------
    seleccionados : list of pandas.DataFrame
        List of spectral DataFrames to be fused at the low level.
    nombres_seleccionados : list of str
        Names or identifiers of the selected datasets, used to label the fused output.
    lista_rangos : list of tuple
        Per-dataset range information (e.g., min and max X values) used to align axes during fusion.
    interseccion : bool
        Whether to restrict fusion to the common intersection of all X-axis ranges.
    rang_comun : tuple
        Common X-axis range shared by all datasets when `interseccion` is True.
    rango_completo : tuple
        Overall X-axis range covering all datasets, used when building a global fusion grid.
    rango_comun : tuple
        Effective X-axis range to use for the fusion grid, which may be the intersection or full range.
    opciones_metodo : dict
        Configuration of resampling or interpolation methods to use when aligning datasets.
    opciones_paso : dict
        Options describing how to determine the X-axis step size (e.g., fixed step or number of points).
    input_paso : any
        User-provided step value associated with `opciones_paso` when a fixed step is requested.
    input_n_puntos : any
        User-provided number of points associated with `opciones_paso` when uniform sampling is requested.
    tipos_orden : list
        Ordered list of sample types or classes used to arrange columns in the fused DataFrame.
    modo_concat : str
        Concatenation mode describing how datasets are appended (e.g., by blocks or interleaved).
    interpolar : bool
        Whether to interpolate spectra onto the chosen fusion grid when exact X values do not match.

    """

    signal_datalowfusion = Signal(object)

    def __init__(
        self,
        seleccionados,
        nombres_seleccionados,
        lista_rangos,
        interseccion,
        rang_comun,
        rango_completo,
        rango_comun,
        opciones_metodo,
        opciones_paso,
        input_paso,
        input_n_puntos,
        tipos_orden,
        modo_concat,
        interpolar,
    ):
        super().__init__()
        self.seleccionados = seleccionados
        self.nombres_seleccionados = nombres_seleccionados
        self.lista_rangos = lista_rangos
        self.interseccion = interseccion
        self.rang_comun = rang_comun
        self.rango_completo = rango_completo
        self.rango_comun = rango_comun
        self.opciones_metodo = opciones_metodo
        self.opciones_paso = opciones_paso
        self.input_paso = input_paso
        self.input_n_puntos = input_n_puntos
        self.tipos_orden = tipos_orden
        self.modo_concat = modo_concat
        self.interpolar = interpolar

    def run(self):

        dataframe_concatenado = concatenate_low_level_fusion(
            self.seleccionados,
            self.nombres_seleccionados,
            self.lista_rangos,
            self.interseccion,
            self.rang_comun,
            self.rango_completo,
            self.rango_comun,
            self.opciones_metodo,
            self.opciones_paso,
            self.input_paso,
            self.input_n_puntos,
            self.tipos_orden,
            self.modo_concat,
            self.interpolar,
        )
        self.signal_datalowfusion.emit(dataframe_concatenado)


class LowLevelDataFusionNoCommonRangeThread(QThread):
    """
    Performs low-level data fusion when spectral datasets do not share a common X-axis range.
    The thread resamples each dataset independently according to the chosen method and emits a concatenated DataFrame without enforcing range intersection.

    Parameters
    ----------
    seleccionados : list of pandas.DataFrame
        Spectral DataFrames to be fused without restricting to a common X-axis range.
    nombres_seleccionados : list of str
        Names or identifiers of the selected datasets, used for labeling or debugging.
    lista_rangos : list of tuple
        Per-dataset range information (e.g., min and max X values) used to guide resampling.
    input_n_puntos : any
        Desired number of points in the resampled spectra grid for each dataset.
    opciones_metodo : dict
        Configuration of resampling or interpolation methods to apply when reconstructing each spectrum.
    tipos_orden : list
        Ordered list of sample types or classes used to arrange columns in the fused DataFrame.
    """

    signal_datalowfusionsininterseccion = Signal(object)

    def __init__(
        self,
        seleccionados,
        nombres_seleccionados,
        lista_rangos,
        input_n_puntos,
        opciones_metodo,
        tipos_orden,
    ):
        super().__init__()
        self.seleccionados = seleccionados
        self.nombres_seleccionados = nombres_seleccionados
        self.lista_rangos = lista_rangos
        self.input_n_puntos = input_n_puntos
        self.opciones_metodo = opciones_metodo
        self.tipos_orden = tipos_orden

    def run(self):

        dataframe_concatenado = concatenate_low_level_fusion_without_intersection(
            self.seleccionados,
            self.input_n_puntos,
            self.opciones_metodo,
            self.tipos_orden,
        )

        self.signal_datalowfusionsininterseccion.emit(dataframe_concatenado)


class MidLevelDataFusionThread(QThread):
    """
    Performs mid-level data fusion by combining feature representations (e.g., PCA scores) from multiple spectral datasets in a background thread.
    The thread concatenates mid-level features according to user-selected ranges and parameters, computes explained variance, and emits both the fused DataFrame and variance list.

    Parameters
    ----------
    seleccionados : list of pandas.DataFrame
        Spectral DataFrames to be fused at the mid level.
    nombres_seleccionados : list of str
        Names or identifiers of the selected datasets, used for labeling and reporting.
    lista_rangos : list of tuple
        Per-dataset range information (e.g., min and max X values) used when aligning axes prior to feature extraction.
    interseccion : bool
        Whether to restrict fusion to the common intersection of all X-axis ranges.
    rang_comun : tuple
        Common X-axis range shared by all datasets when `interseccion` is True.
    rango_completo : tuple
        Overall X-axis range covering all datasets, used when building a global fusion grid.
    rango_comun : tuple
        Effective X-axis range to use for the fusion grid, which may be the intersection or full range.
    opciones_metodo : dict
        Configuration of preprocessing and feature extraction methods applied before fusion.
    opciones_paso : dict
        Options describing how to determine the X-axis step size (e.g., fixed step or number of points) for resampling.
    input_paso : any
        User-provided step value associated with `opciones_paso` when a fixed step is requested.
    input_n_puntos : any
        User-provided number of points associated with `opciones_paso` when uniform sampling is requested.
    tipos_orden : list
        Ordered list of sample types or classes used to arrange rows or columns in the fused feature matrix.
    n_componentes : int or str
        Number of components to retain from each dataset's feature extraction stage.
    intervalo_confianza : int or float
        Confidence interval or opacity scaling factor used later when plotting fused PCA results.
    """

    signal_datamidfusion = Signal(object, object)

    def __init__(
        self,
        seleccionados,
        nombres_seleccionados,
        lista_rangos,
        interseccion,
        rang_comun,
        rango_completo,
        rango_comun,
        opciones_metodo,
        opciones_paso,
        input_paso,
        input_n_puntos,
        tipos_orden,
        n_componentes,
        intervalo_confianza,
    ):
        super().__init__()
        self.seleccionados = seleccionados
        self.nombres_seleccionados = nombres_seleccionados
        self.lista_rangos = lista_rangos
        self.interseccion = interseccion
        self.rang_comun = rang_comun
        self.rango_completo = rango_completo
        self.rango_comun = rango_comun
        self.opciones_metodo = opciones_metodo
        self.opciones_paso = opciones_paso
        self.input_paso = input_paso
        self.input_n_puntos = input_n_puntos
        self.tipos_orden = tipos_orden
        self.intervalo_confianza = intervalo_confianza
        self.n_componentes = n_componentes

    def run(self):
        dataframe_concatenado, lista_varianza = concatenate_mid_level_fusion(
            self.seleccionados,
            self.nombres_seleccionados,
            self.lista_rangos,
            self.interseccion,
            self.rang_comun,
            self.rango_completo,
            self.rango_comun,
            self.opciones_metodo,
            self.opciones_paso,
            self.input_paso,
            self.input_n_puntos,
            self.tipos_orden,
            self.n_componentes,
            self.intervalo_confianza,
        )
        self.signal_datamidfusion.emit(dataframe_concatenado, lista_varianza)


class MidLevelDataFusionNoCommonRangeThread(QThread):
    """
    Performs mid-level data fusion when spectral datasets do not share a common X-axis range.
    The thread extracts and concatenates feature representations from each dataset independently, computes explained variance, and emits both the fused DataFrame and variance list.

    Parameters
    ----------
    seleccionados : list of pandas.DataFrame
        Spectral DataFrames to be fused at the mid level without enforcing a common X-axis range.
    nombres_seleccionados : list of str
        Names or identifiers of the selected datasets, used for labeling and reporting.
    lista_rangos : list of tuple
        Per-dataset range information (e.g., min and max X values) used to guide feature extraction.
    input_n_puntos : any
        Desired number of points in the resampled spectra grid for each dataset before feature extraction.
    opciones_metodo : dict
        Configuration of preprocessing and feature extraction methods applied independently to each dataset.
    tipos_orden : list
        Ordered list of sample types or classes used to arrange rows or columns in the fused feature matrix.
    n_componentes : int or str
        Number of components to retain from each dataset's feature extraction stage.
    intervalo_confianza : int or float
        Confidence interval or opacity scaling factor used later when plotting fused PCA results.
    """

    signal_datamidfusionsininterseccion = Signal(object, object)

    def __init__(
        self,
        seleccionados,
        nombres_seleccionados,
        lista_rangos,
        input_n_puntos,
        opciones_metodo,
        tipos_orden,
        n_componentes,
        intervalo_confianza,
    ):
        super().__init__()
        self.seleccionados = seleccionados
        self.nombres_seleccionados = nombres_seleccionados
        self.lista_rangos = lista_rangos
        self.input_n_puntos = input_n_puntos
        self.opciones_metodo = opciones_metodo
        self.tipos_orden = tipos_orden
        self.intervalo_confianza = intervalo_confianza
        self.n_componentes = n_componentes

    def run(self):

        dataframe_concatenado, lista_varianza = (
            concatenate_mid_level_fusion_without_intersection(
                self.seleccionados,
                self.input_n_puntos,
                self.opciones_metodo,
                self.tipos_orden,
                self.n_componentes,
                self.intervalo_confianza,
            )
        )
        self.signal_datamidfusionsininterseccion.emit(
            dataframe_concatenado, lista_varianza
        )


class MidLevelPlotThread(QThread):
    """
    Generates PCA-based visualization figures for mid-level data fusion results in a background thread.
    The thread creates 2D or 3D score plots and heatmaps for selected components and emits each matplotlib figure via Qt signals for display in the GUI.

    Parameters
    ----------
    lista_df : list of pandas.DataFrame
        List of original spectral DataFrames in the internal format, used to obtain sample types and X-axis information.
    seleccionados : list of int
        Indices or identifiers of the datasets chosen for mid-level visualization.
    df_concat_midfusion : pandas.DataFrame
        DataFrame containing concatenated PCA scores or features from the mid-level fusion step.
    componentes_seleccionados : list of int
        Indices of principal components to visualize in 2D, 3D, or heatmap plots.
    n_componentes : QLineEdit
        GUI input widget holding the number of components to consider for the visualization.
    intervalo_confianza : QLineEdit
        GUI input widget holding the confidence interval or opacity value to use in score plots.
    lista_varianza : list of array-like
        List of explained-variance arrays (one per dataset) that will be flattened into a single variance list.
    """

    pca_2d_figure_signal = Signal(object)
    pca_3d_figure_signal = Signal(object)
    signal_figura_heatmap = Signal(object)

    def __init__(
        self,
        lista_df,
        seleccionados,
        df_concat_midfusion,
        componentes_seleccionados,
        n_componentes,
        intervalo_confianza,
        lista_varianza,
    ):
        super().__init__()
        self.seleccionados = seleccionados
        self.df_concat_midfusion = df_concat_midfusion
        self.componentes_seleccionados = componentes_seleccionados
        self.n_componentes = n_componentes
        self.intervalo_confianza = intervalo_confianza
        self.df = seleccionados[0] if seleccionados else lista_df[0]

        # Accept either legacy QLineEdit widgets or plain Python values.
        if hasattr(self.intervalo_confianza, "text"):
            confidence_value = self.intervalo_confianza.text()
        else:
            confidence_value = self.intervalo_confianza
        self.intervalo_confianza = int(float(confidence_value))

        if isinstance(self.n_componentes, (list, tuple, np.ndarray)):
            self.component_counts = [int(value) for value in self.n_componentes]
        elif hasattr(self.n_componentes, "text"):
            value = int(self.n_componentes.text())
            self.component_counts = [value] * len(lista_varianza)
        else:
            value = int(self.n_componentes)
            self.component_counts = [value] * len(lista_varianza)

        self.lista_varianza = lista_varianza

        stored_labels = getattr(
            df_concat_midfusion,
            "attrs",
            {},
        ).get("sample_labels")

        if stored_labels is not None:
            self.tipos = pd.Series([str(value).strip() for value in stored_labels])
        else:
            self.tipos = self.df.iloc[0, 1:].astype(str).str.strip()

        if len(self.tipos) != len(df_concat_midfusion):
            raise ValueError(
                "The number of class labels does not match the number "
                "of fused samples."
            )

        tipos_nombres = self.tipos.unique()
        cmap = plt.cm.Spectral
        colores = [cmap(i) for i in np.linspace(0, 1, len(tipos_nombres))]
        self.color_mapping = {
            tipo: mcolors.to_hex(colores[i]) for i, tipo in enumerate(tipos_nombres)
        }
        self.raman_shift = self.df.iloc[1:, 0].reset_index(drop=True)

        lista_varianza_unificada = np.concatenate(lista_varianza).tolist()
        self.lista_varianza = lista_varianza_unificada

        max_index = len(self.lista_varianza)
        if any(i > max_index for i in componentes_seleccionados):
            print(
                "[ERROR] At least one of the selected components is outside the available variance range."
            )
            return

        self.pca_resultado = df_concat_midfusion.copy()
        self.explained_variance_percentage = self.lista_varianza

    def run(self):
        if len(self.componentes_seleccionados) == 2:
            if len(self.componentes_seleccionados) == 2:
                pc_x, pc_y = self.componentes_seleccionados
            else:
                return
            dato_pca_array = self.pca_resultado.to_numpy()
            fig = plot_pca_2d(
                dato_pca_array,
                self.explained_variance_percentage,
                self.color_mapping,
                self.tipos,
                pc_x,
                pc_y,
                self.intervalo_confianza,
            )
            self.pca_2d_figure_signal.emit(fig)

        if len(self.componentes_seleccionados) == 3:
            pc_x, pc_y, pc_z = self.componentes_seleccionados

            dato_pca_array = self.pca_resultado.to_numpy()
            fig = plot_pca_3d(
                dato_pca_array,
                self.explained_variance_percentage,
                self.color_mapping,
                self.tipos,
                pc_x,
                pc_y,
                pc_z,
                self.intervalo_confianza,
            )
            self.pca_3d_figure_signal.emit(fig)

        if len(self.componentes_seleccionados) > 3:
            dato_pca_array = self.pca_resultado.to_numpy()
            tipos_alineados = self.tipos.reset_index(drop=True).iloc[
                : dato_pca_array.shape[0]
            ]
            fig = plot_heatmap_pca(
                dato_pca_array, tipos_alineados, self.componentes_seleccionados
            )
            self.signal_figura_heatmap.emit(fig)