import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.cluster.hierarchy as sch
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.figure_factory as ff
import re
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import fcluster
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.ticker import MaxNLocator

from core.translations import translate, get_language


def tr(text, **values):
    return translate(text, get_language(), **values)


def plot_heatmap_pca(dato_pca, tipos, componentes_seleccionados):
    """
    Plot a heatmap of selected principal components with samples ordered by type. This helps visualize how different types cluster or differ in PCA space.

    The function builds a DataFrame from the PCA scores, filters it to the requested components, and aligns it with the provided type labels, truncating either the labels or the data if their lengths differ. It then sorts samples by type, draws a Seaborn heatmap of the selected components, and returns the resulting Matplotlib Figure.

    Parameters
    ----------
    dato_pca : array-like of shape (n_samples, n_components)
        PCA score matrix where rows are samples and columns are principal
        components.
    tipos : array-like
        Sequence of type or class labels corresponding to each sample in
        `dato_pca`. It may be longer or shorter than the number of samples and
        will be truncated or matched accordingly.
    componentes_seleccionados : iterable of int
        Iterable of principal component indices (1-based) to include in the
        heatmap; only components present in `dato_pca` are used.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib Figure object containing the heatmap of the selected
        principal components, with samples ordered by their type labels.
    """
    df_pca = pd.DataFrame(
        dato_pca, columns=[f"PC{i+1}" for i in range(dato_pca.shape[1])]
    )
    columnas_usar = [
        f"PC{i}" for i in componentes_seleccionados if f"PC{i}" in df_pca.columns
    ]
    df_filtrado = df_pca[columnas_usar].copy()
    n_filas = df_filtrado.shape[0]
    tipos_series = pd.Series(tipos).reset_index(drop=True)
    if len(tipos_series) > n_filas:
        print("⚠️ 'tipos' has more values than rows. It will be truncated.")
        tipos_series = tipos_series.iloc[:n_filas]
    elif len(tipos_series) < n_filas:
        print("⚠️ 'tipos' has fewer values than rows. df_filtrado will be truncated.")
        df_filtrado = df_filtrado.iloc[: len(tipos_series)]

    df_filtrado["Type"] = tipos_series.values

    df_filtrado = df_filtrado.sort_values(by="Type").reset_index(drop=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_filtrado.drop(columns="Type"), cmap="coolwarm", yticklabels=False)
    plt.title("Principal Components Heatmap")
    plt.xlabel("Principal Components")
    plt.ylabel("Samples ordered by type")

    return plt.gcf()


def calculate_cumulative_variance(df, umbral=95):
    """
    Calculates:
    - individual explained variance (%)
    - cumulative explained variance (%)
    - n_threshold: minimum number of PCs required to reach the threshold
    """
    X = prepare_pca_matrix(df)

    modelo_pca = PCA()
    modelo_pca.fit(X)

    var_ind = modelo_pca.explained_variance_ratio_ * 100
    var_acum = np.cumsum(var_ind)
    n_umbral = int(np.argmax(var_acum >= umbral) + 1)

    return var_ind, var_acum, n_umbral


def prepare_pca_matrix(df):
    """
    Prepara la matriz numérica para PCA desde el formato
    interno de EspectroApp.

    Las columnas representan muestras y las filas representan
    variables espectrales.
    """
    if df is None or df.empty:
        raise ValueError("The selected DataFrame is empty.")

    # Convertir la primera columna para detectar solamente
    # las filas que contienen valores reales del eje X.
    x_numeric = pd.to_numeric(
        df.iloc[:, 0],
        errors="coerce",
    )

    valid_rows = x_numeric.notna()

    if not valid_rows.any():
        raise ValueError("No numeric spectral-axis values were found.")

    # Tomar las intensidades únicamente en filas
    # donde el eje X es numérico. Así se eliminan
    # encabezados repetidos como 'Raman Shift'.
    intensity_matrix = df.loc[valid_rows, df.columns[1:]].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Eliminar columnas completamente vacías.
    intensity_matrix = intensity_matrix.dropna(
        axis=1,
        how="all",
    )

    # Comprobar NaN parciales reales.
    columns_with_nan = intensity_matrix.columns[intensity_matrix.isna().any()]

    if len(columns_with_nan) > 0:
        raise ValueError(
            "Some spectra contain non-numeric or missing "
            "intensity values after removing header rows."
        )

    if intensity_matrix.shape[1] < 2:
        raise ValueError("At least two valid spectra are required " "to perform PCA.")

    X = intensity_matrix.T.to_numpy(dtype=float)

    return X


def assign_type_colors(tipos):
    """
    Assigns consistent dark colors to sample types.

    The assignment is generic:
    - It does not depend on specific sample names.
    - It sorts the detected types alphabetically.
    - Therefore, the same set of sample types receives the same colors
      even if the column order changes.
    """

    paleta_colores_oscuros = [
        "#1f77b4",  # blue
        "#d55e00",  # orange/red
        "#009e73",  # green
        "#7b3294",  # purple
        "#4d4d4d",  # dark gray
        "#0072b2",  # dark cyan/blue
        "#a6761d",  # brown
        "#e7298a",  # magenta
        "#66a61e",  # green
        "#000000",  # black
    ]

    tipos_series = pd.Series(tipos).dropna()
    tipos_unicos = tipos_series.unique()
    tipos_ordenados = sorted(tipos_unicos, key=lambda x: str(x).strip().lower())

    asignacion = {
        tipo: paleta_colores_oscuros[i % len(paleta_colores_oscuros)]
        for i, tipo in enumerate(tipos_ordenados)
    }

    return asignacion
