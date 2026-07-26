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
from .utilities import prepare_pca_matrix

from core.translations import translate, get_language


def tr(text, **values):
    return translate(text, get_language(), **values)


def plot_loadings(X, raman_shift, op_pca):
    """
    Plot PCA loading vectors against the spectral axis for selected components.
    op_pca uses 1-based indexing: PC1, PC2, PC3...
    """

    pcs = []
    for pc in op_pca:
        try:
            pc = int(pc)
            if pc > 0:
                pcs.append(pc)
        except Exception:
            continue

    pcs = sorted(set(pcs))

    if not pcs:
        raise ValueError(
            "No valid principal components were selected for the loading plot."
        )

    X = np.asarray(X, dtype=float)
    x = pd.to_numeric(pd.Series(raman_shift).reset_index(drop=True), errors="coerce")

    valid_x = x.notna().to_numpy()
    x = x[valid_x].to_numpy(dtype=float)

    if X.shape[1] == len(valid_x):
        X = X[:, valid_x]
    else:
        raise ValueError(
            f"X-axis length does not match PCA variables: "
            f"len(raman_shift)={len(valid_x)}, X.shape={X.shape}"
        )

    max_pc = min(X.shape[0], X.shape[1])

    if max(pcs) > max_pc:
        raise ValueError(
            f"The selected PC exceeds the maximum available PC. "
            f"Selected PC{max(pcs)}, but max is PC{max_pc}."
        )

    modelo_pca = PCA(n_components=max(pcs))
    modelo_pca.fit(X)

    loadings = modelo_pca.components_
    varianza = modelo_pca.explained_variance_ratio_ * 100

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    for pc in pcs:
        idx = pc - 1
        ax.plot(x, loadings[idx], label=f"PC{pc} ({varianza[idx]:.2f}%)")

    ax.axhline(0, color="black", linewidth=1, linestyle="--")

    axis_label = "Spectral variable"
    if len(x) and np.nanmax(x) <= 5000:
        axis_label = "Wavenumber / Raman shift (cm⁻¹)"
    ax.set_xlabel(axis_label)
    ax.set_ylabel(tr("Loading"))
    ax.set_title(tr("PCA Loading Plot"))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.legend()
    ax.grid(True)

    return fig


def infer_spectral_block_name(name, df, index):
    """
    Infers a readable block name for a low-level fusion block.
    """
    text = str(name).lower()

    try:
        first_cell = str(df.iloc[0, 0]).lower()
    except Exception:
        first_cell = ""

    combined_text = f"{text} {first_cell}"

    if "raman" in combined_text or "shift" in combined_text:
        return "Raman"

    if (
        "ftir" in combined_text
        or "ft-ir" in combined_text
        or "infrared" in combined_text
        or "ir" in combined_text
        or "wavenumber" in combined_text
        or "wave number" in combined_text
        or "número de onda" in combined_text
        or "numero de onda" in combined_text
    ):
        return "FTIR"

    return f"Block {index + 1}"


def add_low_level_fusion_metadata(df_fusion, selected_dfs, selected_names=None):
    """
    Adds metadata to a low-level fused DataFrame indicating where each
    original spectral block starts and ends in the fused matrix.

    The start/end positions are relative to the PCA variable axis,
    i.e., relative to df_fusion.iloc[1:, 1:].
    """
    if selected_names is None:
        selected_names = [f"Block {i + 1}" for i in range(len(selected_dfs))]

    blocks = []
    start = 0

    for i, df in enumerate(selected_dfs):
        df_tmp = df.reset_index(drop=True).copy()

        x = pd.to_numeric(df_tmp.iloc[:, 0], errors="coerce")
        valid_x = x.notna()
        x_values = x[valid_x].to_numpy(dtype=float)

        n_variables = len(x_values)
        end = start + n_variables

        block_name = infer_spectral_block_name(
            selected_names[i] if i < len(selected_names) else f"Block {i + 1}",
            df_tmp,
            i,
        )

        blocks.append(
            {
                "name": block_name,
                "source_name": (
                    str(selected_names[i]) if i < len(selected_names) else block_name
                ),
                "start": start,
                "end": end,
                "x": x_values.tolist(),
            }
        )

        start = end

    df_fusion.attrs["fusion_type"] = "low_level"
    df_fusion.attrs["fusion_blocks"] = blocks

    return df_fusion


def plot_low_level_fusion_loadings(df_fusion, op_pca):
    """
    Plots PCA loadings from a low-level fused matrix, separating the
    loading vectors by the original spectral blocks.

    PCA is calculated using the complete fused matrix. Only after PCA
    the loading vectors are divided into FTIR/Raman blocks.
    """
    if df_fusion.attrs.get("fusion_type") != "low_level":
        raise ValueError("The selected DataFrame is not marked as low-level fusion.")

    blocks = df_fusion.attrs.get("fusion_blocks", None)

    if not blocks:
        raise ValueError("No low-level fusion block metadata was found.")

    pcs = []
    for pc in op_pca:
        try:
            pc = int(pc)
            if pc > 0:
                pcs.append(pc)
        except Exception:
            continue

    pcs = sorted(set(pcs))

    if not pcs:
        raise ValueError("No valid principal components were selected.")

    X = prepare_pca_matrix(df_fusion)

    max_pc = min(X.shape[0], X.shape[1])

    if max(pcs) > max_pc:
        raise ValueError(
            f"The selected PC exceeds the maximum available PC. "
            f"Selected PC{max(pcs)}, but max is PC{max_pc}."
        )

    modelo_pca = PCA(n_components=max(pcs))
    modelo_pca.fit(X)

    loadings = modelo_pca.components_
    varianza = modelo_pca.explained_variance_ratio_ * 100

    fig = Figure(figsize=(11, 4 * len(blocks)))

    for i, block in enumerate(blocks, start=1):
        ax = fig.add_subplot(len(blocks), 1, i)

        name = str(block["name"]).strip()
        name_lower = name.lower()

        # Normalizar nombre del bloque para mostrarlo bien en el gráfico
        if "raman" in name_lower:
            display_name = "Raman"
            x_label = "Raman shift (cm$^{-1}$)"
        elif "ftir" in name_lower or "ft-ir" in name_lower or "infrared" in name_lower:
            display_name = "FTIR"
            x_label = "FTIR wavenumber (cm$^{-1}$)"
        else:
            display_name = name
            x_label = f"{display_name} spectral variable (cm$^{-1}$)"

        start = int(block["start"])
        end = int(block["end"])
        x = np.asarray(block["x"], dtype=float)

        for pc in pcs:
            idx = pc - 1
            y = loadings[idx, start:end]

            if len(x) != len(y):
                raise ValueError(
                    f"Length mismatch in {display_name}: x={len(x)}, loading={len(y)}"
                )

            ax.plot(x, y, label=f"PC{pc} ({varianza[idx]:.2f}%)")

        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(f"PCA loading plot - {display_name}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(tr("Loading"))

        ax.xaxis.set_major_locator(MaxNLocator(nbins=9))

        # Eje Y automático para que FTIR y Raman tengan marcas legibles
        y_min, y_max = ax.get_ylim()
        y_abs = max(abs(y_min), abs(y_max))

        if y_abs < 0.01:
            y_format = "%.4f"
        elif y_abs < 0.1:
            y_format = "%.3f"
        else:
            y_format = "%.2f"

        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        y_min, y_max = ax.get_ylim()
        y_abs = max(abs(y_min), abs(y_max))

        if y_abs < 1e-3:
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)

        elif y_abs < 0.1:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    return fig
