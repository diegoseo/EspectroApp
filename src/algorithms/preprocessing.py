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


def get_column_with_fewest_rows(df):
    """Determine the column with the fewest non-null values in a DataFrame. Returns both the column name and the corresponding non-null count.

    The function scans all columns, counts their non-null entries, and identifies the one with the smallest count. This is useful for quickly locating the sparsest column in terms of available data.

    Args:
        df: Input pandas DataFrame to analyze.

    Returns:
        A tuple containing:
            - The name of the column with the fewest non-null values.
            - The number of non-null values in that column.
    """
    valores_no_nulos = df.notna().sum()

    columna_menor = valores_no_nulos.idxmin()
    cantidad_menor = valores_no_nulos.min()

    return columna_menor, cantidad_menor


def normalize_by_mean(df, metodo):
    numeric_df = df.apply(pd.to_numeric, errors="coerce").astype(float).copy()

    if numeric_df.isna().any().any():
        raise ValueError(
            "Normalization requires numeric data " "without missing values."
        )

    if metodo == "Standardize u=0, v2=1":
        df_transpuesta = numeric_df.T

        mean_values = df_transpuesta.mean(axis=0)
        std_values = df_transpuesta.std(axis=0, ddof=1)
        std_values = std_values.replace(0, 1)

        resultado = (df_transpuesta - mean_values) / std_values

        return resultado.T

    elif metodo == "Center to u=0":
        df_transpuesta = numeric_df.T

        resultado = df_transpuesta - df_transpuesta.mean(axis=0)

        return resultado.T

    elif metodo == "Scale to v2=1":
        df_transpuesta = numeric_df.T

        std_values = df_transpuesta.std(
            axis=0,
            ddof=1,
        ).replace(0, 1)

        resultado = df_transpuesta / std_values

        return resultado.T

    elif metodo == "Normalize to interval [-1,1]":
        min_values = numeric_df.min(axis=0)
        max_values = numeric_df.max(axis=0)

        ranges = (max_values - min_values).replace(0, 1)

        return 2 * ((numeric_df - min_values) / ranges) - 1

    elif metodo == "Normalize to interval [0,1]":
        min_values = numeric_df.min(axis=0)
        max_values = numeric_df.max(axis=0)

        ranges = (max_values - min_values).replace(0, 1)

        return (numeric_df - min_values) / ranges

    raise ValueError(f"Unsupported normalization method: {metodo}")


def normalize_by_area(df, raman_shift):
    """
    Normalize each spectrum in a DataFrame by the area under its curve. This produces spectra that are comparable in overall intensity regardless of their original magnitude.

    The function computes the numerical integral of each column with respect to the Raman shift and scales the values so that each spectrum has unit area (up to a sign). Columns with zero area are left unchanged to avoid division by zero.

    Parameters
    ----------
    df : pandas.DataFrame
        Matrix of spectral intensities, where each column represents a spectrum
        and rows correspond to Raman shift positions.
    raman_shift : pandas.Series or array-like
        Ordered Raman shift values used as the x-axis for numerical integration.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same shape as `df`, where each column has been
        normalized by its integrated area.
    """
    columnas_normalizadas = []
    np_array = raman_shift.to_numpy()
    for col in df.columns:
        y = df[col].to_numpy()
        area = np.trapezoid(y, np_array) * -1
        if area != 0:
            normalizado = y / area
        else:
            normalizado = y

        columnas_normalizadas.append(pd.Series(normalizado, name=col))

    df_normalizado = pd.concat(columnas_normalizadas, axis=1)
    return df_normalizado


def smooth_savitzky_golay(df, ventana, orden):
    """
    Smooth spectral or time series data using the Savitzky-Golay filter. This reduces noise while preserving peak shapes and important signal features.

    The function applies the Savitzky-Golay filter independently to each column of the input DataFrame, returning a new DataFrame with the smoothed values. It is intended for preprocessing spectral or similar structured numeric data prior to analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data where each column represents a signal to be smoothed and rows
        represent ordered measurement points.
    ventana : int
        Window length (number of points) used by the Savitzky-Golay filter; must
        be a positive odd integer greater than the polynomial order.
    orden : int
        Polynomial order used to fit the samples within each window.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same shape and columns as `df`, containing the
        smoothed signals.
    """

    dato = df.to_numpy()

    suavizado = np.apply_along_axis(
        lambda x: savgol_filter(x, window_length=ventana, polyorder=orden),
        axis=0,
        arr=dato,
    )

    suavizado_df = pd.DataFrame(suavizado, columns=df.columns)

    return suavizado_df


def smooth_gaussian_filter(df, sigma):
    """
    Smooth numerical signals in a DataFrame using a Gaussian filter. This reduces high-frequency noise while preserving the overall shape of each signal.

    The function applies a one-dimensional Gaussian convolution independently to each column of the input DataFrame and returns the smoothed result. It is suitable for preprocessing spectral or time-series data prior to further analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data where each column represents a signal and rows represent
        ordered measurement points.
    sigma : float
        Standard deviation of the Gaussian kernel controlling the degree of
        smoothing; larger values produce smoother signals.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same shape and columns as `df`, containing the
        Gaussian-smoothed signals.
    """
    dato = df.to_numpy(dtype=float)
    suavizado_gaussiano = np.apply_along_axis(
        lambda x: gaussian_filter1d(x, sigma=sigma), axis=0, arr=dato
    )
    suavizado_gaussiano_pd = pd.DataFrame(suavizado_gaussiano, columns=df.columns)
    return suavizado_gaussiano_pd


def smooth_moving_average(df, ventana):
    """
    Smooth DataFrame columns using a centered moving average. This reduces short-term fluctuations while retaining the overall trend of each signal.

    The function applies a rolling mean with the specified window size independently to each column, centering the window around each point and allowing partial windows at the edges. It is suitable for denoising spectral, time-series, or similar sequential data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data where each column represents a signal and rows correspond to
        ordered measurement points.
    ventana : int
        Size of the rolling window used to compute the moving average.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same shape and columns as `df`, containing the
        moving-average smoothed signals.
    """
    suavizado_media_movil = df.rolling(
        window=ventana, min_periods=1, center=True
    ).mean()

    return suavizado_media_movil


def linear_baseline_from_points(x, y, x_start, x_end):
    """Build and subtract a straight baseline defined by two X positions.

    The Y coordinate of each anchor is taken from the nearest real point of the
    spectrum. Returns both the corrected signal and the estimated baseline.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if x.size != y.size:
        raise ValueError("X-axis and spectrum must have the same length.")
    if x.size < 2:
        raise ValueError("At least two spectral points are required.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Baseline correction requires finite numeric values.")

    x_start = float(x_start)
    x_end = float(x_end)
    if np.isclose(x_start, x_end):
        raise ValueError("The two baseline X positions must be different.")

    idx_start = int(np.argmin(np.abs(x - x_start)))
    idx_end = int(np.argmin(np.abs(x - x_end)))

    x1, y1 = float(x[idx_start]), float(y[idx_start])
    x2, y2 = float(x[idx_end]), float(y[idx_end])

    if np.isclose(x1, x2):
        raise ValueError("The selected baseline points map to the same X value.")

    slope = (y2 - y1) / (x2 - x1)
    baseline = y1 + slope * (x - x1)
    corrected = y - baseline
    return corrected, baseline


def correct_linear_baseline(df, raman_shift, x_start=None, x_end=None):
    """Apply point-defined linear baseline correction to every spectrum.

    The same two X positions are used for all spectra, while each spectrum uses
    its own Y values at those positions. Therefore, every column receives an
    individual straight baseline.
    """
    if x_start is None or x_end is None:
        raise ValueError("Linear baseline correction requires x_start and x_end.")

    x = np.asarray(raman_shift, dtype=float).reshape(-1)
    intensities = df.apply(pd.to_numeric, errors="coerce")

    if len(x) != len(intensities):
        raise ValueError("X-axis length does not match the intensity matrix.")
    if intensities.isna().any().any():
        raise ValueError(
            "Linear baseline correction cannot be applied while intensities "
            "contain missing or non-numeric values."
        )

    corrected_columns = {}
    for column in intensities.columns:
        corrected, _ = linear_baseline_from_points(
            x,
            intensities[column].to_numpy(dtype=float),
            x_start,
            x_end,
        )
        corrected_columns[column] = corrected

    return pd.DataFrame(corrected_columns, columns=intensities.columns)


def linear_baseline_correction(y, raman_shift):
    """Legacy end-to-end linear correction kept for compatibility."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(raman_shift, dtype=float)
    corrected, _ = linear_baseline_from_points(x, y, x[0], x[-1])
    return corrected


def shirley_baseline_from_points(
    x,
    y,
    x_start,
    x_end,
    tolerance=1e-6,
    max_iterations=100,
    return_info=False,
):
    """Estimate and subtract an iterative Shirley baseline inside two X limits.

    The selected X limits are snapped to the nearest real spectral positions.
    The baseline is held constant outside the selected interval so that the
    corrected signal keeps the original length.

    Parameters
    ----------
    x, y : array-like
        Spectral axis and one spectrum.
    x_start, x_end : float
        User-selected limits of the Shirley region.
    tolerance : float
        Relative convergence tolerance.
    max_iterations : int
        Maximum number of Shirley updates.
    return_info : bool
        When True, also return a dictionary with convergence information.

    Returns
    -------
    corrected, baseline : numpy.ndarray
        Corrected signal and estimated baseline in the original X order.
    info : dict, optional
        Returned only when ``return_info=True``.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if x.size != y.size:
        raise ValueError("X-axis and spectrum must have the same length.")
    if x.size < 3:
        raise ValueError("Shirley correction requires at least three points.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Shirley correction requires finite numeric values.")

    tolerance = float(tolerance)
    max_iterations = int(max_iterations)

    if tolerance <= 0:
        raise ValueError("Shirley tolerance must be greater than zero.")
    if max_iterations < 1:
        raise ValueError("Shirley maximum iterations must be at least 1.")

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    idx_1 = int(np.argmin(np.abs(x_sorted - float(x_start))))
    idx_2 = int(np.argmin(np.abs(x_sorted - float(x_end))))
    left_idx, right_idx = sorted((idx_1, idx_2))

    if left_idx == right_idx:
        raise ValueError("The two Shirley limits map to the same X value.")
    if right_idx - left_idx < 2:
        raise ValueError(
            "The Shirley interval must contain at least three spectral points."
        )

    xr = x_sorted[left_idx : right_idx + 1]
    yr = y_sorted[left_idx : right_idx + 1]

    y_left = float(yr[0])
    y_right = float(yr[-1])

    # Stable initial estimate joining the selected endpoints.
    baseline_region = np.linspace(y_left, y_right, yr.size, dtype=float)

    converged = False
    completed_iterations = 0
    signal_scale = max(float(np.ptp(yr)), float(np.max(np.abs(yr))), 1.0)

    for iteration in range(1, max_iterations + 1):
        residual = np.clip(yr - baseline_region, 0.0, None)

        # Trapezoidal area accumulated from each point to the right endpoint.
        segment_area = 0.5 * (residual[:-1] + residual[1:]) * np.diff(xr)
        area_to_right = np.zeros_like(yr, dtype=float)
        area_to_right[:-1] = np.cumsum(segment_area[::-1])[::-1]
        total_area = float(area_to_right[0])

        if not np.isfinite(total_area) or total_area <= np.finfo(float).eps:
            # No usable positive area remains; endpoint line is the stable result.
            new_baseline = np.linspace(y_left, y_right, yr.size, dtype=float)
        else:
            fraction = area_to_right / total_area
            new_baseline = y_right + (y_left - y_right) * fraction
            new_baseline[0] = y_left
            new_baseline[-1] = y_right

        difference = float(np.max(np.abs(new_baseline - baseline_region)))
        baseline_region = new_baseline
        completed_iterations = iteration

        if difference <= tolerance * signal_scale:
            converged = True
            break

    baseline_sorted = np.empty_like(y_sorted, dtype=float)
    baseline_sorted[:left_idx] = baseline_region[0]
    baseline_sorted[left_idx : right_idx + 1] = baseline_region
    baseline_sorted[right_idx + 1 :] = baseline_region[-1]

    corrected_sorted = y_sorted - baseline_sorted

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)
    baseline = baseline_sorted[inverse_order]
    corrected = corrected_sorted[inverse_order]

    info = {
        "iterations": completed_iterations,
        "converged": converged,
        "x_start": float(x_sorted[left_idx]),
        "x_end": float(x_sorted[right_idx]),
    }

    if return_info:
        return corrected, baseline, info
    return corrected, baseline


def correct_shirley_baseline(
    df,
    raman_shift,
    x_start=None,
    x_end=None,
    tolerance=1e-6,
    max_iterations=100,
):
    """Apply an iterative Shirley correction independently to every spectrum."""
    if x_start is None or x_end is None:
        raise ValueError("Shirley baseline correction requires x_start and x_end.")

    x = np.asarray(raman_shift, dtype=float).reshape(-1)
    intensities = df.apply(pd.to_numeric, errors="coerce")

    if len(x) != len(intensities):
        raise ValueError("X-axis length does not match the intensity matrix.")
    if intensities.isna().any().any():
        raise ValueError(
            "Shirley correction cannot be applied while intensities "
            "contain missing or non-numeric values."
        )

    corrected_columns = {}
    for column in intensities.columns:
        corrected, _ = shirley_baseline_from_points(
            x,
            intensities[column].to_numpy(dtype=float),
            x_start=x_start,
            x_end=x_end,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        corrected_columns[column] = corrected

    return pd.DataFrame(corrected_columns, columns=intensities.columns)


def calculate_first_derivative(df, raman_shift):
    """
    Compute the first numerical derivative of each spectrum with respect to the spectral axis. This highlights changes in signal intensity and can enhance subtle features in spectral data.

    The function treats each column of the input DataFrame as an independent spectrum and uses `numpy.gradient` to approximate its derivative relative to the provided Raman shift values. The result is returned as a DataFrame aligned with the original index and columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Spectral intensity matrix where each column represents a spectrum and
        rows correspond to ordered Raman shift positions.
    raman_shift : array-like
        One-dimensional sequence of spectral axis values associated with the
        rows of `df`. Must be the same length as the number of rows in `df`.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the first derivative of each input spectrum, with
        the same index and columns as `df`.
    """
    df_derivada = pd.DataFrame(index=df.index, columns=df.columns)

    for col in df.columns:
        y = df[col].values
        primer_der = np.gradient(y, raman_shift)  # Derivada de y respecto a x
        df_derivada[col] = primer_der

    return df_derivada


def calculate_second_derivative(df, raman_shift):
    """
    Compute the second numerical derivative of each spectrum with respect to the spectral axis. This emphasizes curvature and inflection points, which can help reveal subtle features in spectral data.

    The function treats each column of the input DataFrame as an independent spectrum and first approximates its first derivative, then its second derivative, using `numpy.gradient` with the provided Raman shift values. The result is returned as a DataFrame aligned with the original index and columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Spectral intensity matrix where each column represents a spectrum and
        rows correspond to ordered Raman shift positions.
    raman_shift : array-like
        One-dimensional sequence of spectral axis values associated with the
        rows of `df`. Must be the same length as the number of rows in `df`.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the second derivative of each input spectrum, with
        the same index and columns as `df`.
    """
    df_derivada2 = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        y = df[col].values
        primer_der = np.gradient(y, raman_shift)  # First derivative
        segundo_der = np.gradient(primer_der, raman_shift)  # Second derivative
        df_derivada2[col] = segundo_der

    return df_derivada2
