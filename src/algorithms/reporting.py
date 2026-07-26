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


def _safe_filename(name: str) -> str:
    """
    Convert a user-provided report name into a safe file name.
    """
    name = str(name).strip()
    name = re.sub(r"[^\w\s.-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name or "dimensionality_reduction_report"


def _format_sequence(values):
    """
    Format lists, tuples, numpy arrays, or simple values in a readable way.
    """
    if values is None:
        return tr("None")

    if isinstance(values, (list, tuple, set)):
        return ", ".join(str(v) for v in values)

    if isinstance(values, np.ndarray):
        return np.array2string(values, precision=4, suppress_small=True)

    return str(values)


def _format_array_table(array, columns=None, index_prefix="M"):
    """
    Format numpy arrays or pandas DataFrames as readable text tables.
    """
    if array is None:
        return tr("None")

    if isinstance(array, pd.DataFrame):
        return array.to_string(index=True)

    array = np.asarray(array)

    if array.ndim == 1:
        df = pd.DataFrame(array, columns=[tr("Value")])
    elif array.ndim == 2:
        if columns is None:
            columns = [f"Dim_{i+1}" for i in range(array.shape[1])]
        df = pd.DataFrame(
            array,
            columns=columns,
            index=[f"{index_prefix}{i+1}" for i in range(array.shape[0])],
        )
    else:
        return np.array2string(array, precision=4, suppress_small=True)

    return df.round(6).to_string(index=True)


def _format_variance_table(varianza_porcentaje):
    """
    Create a readable PCA variance table with individual and cumulative variance.
    """
    if varianza_porcentaje is None:
        return tr("No PCA variance information available.")

    variance = np.asarray(varianza_porcentaje, dtype=float).flatten()
    cumulative = np.cumsum(variance)

    df = pd.DataFrame(
        {
            tr("Component"): [f"PC{i+1}" for i in range(len(variance))],
            tr("Variance (%)"): variance,
            tr("Cumulative variance (%)"): cumulative,
        }
    )

    return df.round(4).to_string(index=False)


def generate_report(
    nombre_informe,
    opciones,
    componentes,
    intervalo,
    cp_pca,
    cp_tsne,
    componentes_seleccionados,
    asignacion_colores,
    pca_resultado,
    varianza_porcentaje,
    tsne_resultado,
    tsne_pca_resultado,
    output_dir=".",
    dataset_name=None,
    tsne_parameters=None,
):
    """
    Generate a plain-text report summarizing dimensionality reduction settings and results.

    This report includes:
    - general configuration parameters;
    - selected visualization components;
    - confidence interval information;
    - color assignment by class/type;
    - PCA variance explained and cumulative variance;
    - PCA, t-SNE, and t-SNE(PCA(X)) result matrices, when available.

    Parameters
    ----------
    nombre_informe : str
        Base name for the report file.
    opciones : dict or Any
        Enabled options or settings related to preprocessing and analysis.
    componentes : Any
        Principal components used or requested.
    intervalo : Any
        Confidence interval specification.
    cp_pca : int
        Number of PCA components used in t-SNE(PCA(X)).
    cp_tsne : int
        Number of t-SNE components used in t-SNE(PCA(X)).
    componentes_seleccionados : Any
        Components selected for visualization.
    asignacion_colores : dict
        Mapping from class/type labels to assigned colors.
    pca_resultado : Any or None
        PCA scores/result matrix.
    varianza_porcentaje : Any
        Explained variance percentage from PCA.
    tsne_resultado : Any or None
        t-SNE result matrix.
    tsne_pca_resultado : Any or None
        t-SNE(PCA(X)) result matrix.
    output_dir : str or Path, optional
        Folder where the report will be saved.
    dataset_name : str or None, optional
        Optional dataset name to include in the report.

    Returns
    -------
    str
        Path of the generated report file.
    """

    tsne_parameters = tsne_parameters or {}

    safe_name = _safe_filename(nombre_informe)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{safe_name}.txt"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(tr("DIMENSIONALITY REDUCTION REPORT") + "\n")
        f.write("=" * 70 + "\n\n")

        f.write(tr("1. REPORT INFORMATION") + "\n")
        f.write("-" * 70 + "\n")
        f.write(tr("Report name: {name}", name=nombre_informe) + "\n")
        f.write(tr("Generated on: {timestamp}", timestamp=timestamp) + "\n")

        if dataset_name is not None:
            f.write(tr("Dataset: {dataset}", dataset=dataset_name) + "\n")

        f.write("\n")

        f.write(tr("2. GENERAL PARAMETERS") + "\n")
        f.write("-" * 70 + "\n")
        f.write(
            tr(
                "Selected components for visualization: {components}",
                components=_format_sequence(componentes_seleccionados),
            )
            + "\n"
        )
        f.write(f"Confidence interval: {intervalo}\n")
        f.write(f"Principal components requested: {_format_sequence(componentes)}\n")
        f.write(f"Components for PCA in t-SNE(PCA(X)): {cp_pca}\n")
        f.write(f"Components for t-SNE in t-SNE(PCA(X)): {cp_tsne}\n")
        f.write(
            "Direct t-SNE dimensions: " f"{tsne_parameters.get('direct_dimensions')}\n"
        )
        f.write(
            "Direct t-SNE perplexity: " f"{tsne_parameters.get('direct_perplexity')}\n"
        )
        f.write(
            "Direct t-SNE iterations: " f"{tsne_parameters.get('direct_iterations')}\n"
        )
        f.write("PCA+t-SNE perplexity: " f"{tsne_parameters.get('pca_perplexity')}\n")
        f.write("PCA+t-SNE iterations: " f"{tsne_parameters.get('pca_iterations')}\n")

        f.write("\n")

        f.write("3. ENABLED OPTIONS / PREPROCESSING SETTINGS\n")
        f.write("-" * 70 + "\n")

        if isinstance(opciones, dict):
            for key, value in opciones.items():
                f.write(tr("{option_name}: {value}", option_name=key, value=value) + "\n")
        else:
            f.write(f"{opciones}\n")

        f.write("\n")

        f.write("4. COLOR ASSIGNMENT BY TYPE / CLASS\n")
        f.write("-" * 70 + "\n")

        if asignacion_colores:
            for tipo, color in asignacion_colores.items():
                f.write(f"{tipo}: {color}\n")
        else:
            f.write("No color assignment available.\n")

        f.write("\n")

        f.write("5. PCA VARIANCE EXPLAINED\n")
        f.write("-" * 70 + "\n")
        f.write(_format_variance_table(varianza_porcentaje))
        f.write("\n\n")

        if pca_resultado is not None:
            f.write("6. PCA SCORES / PCA RESULT\n")
            f.write("-" * 70 + "\n")
            pca_array = np.asarray(pca_resultado)
            pca_columns = (
                [f"PC{i+1}" for i in range(pca_array.shape[1])]
                if pca_array.ndim == 2
                else None
            )
            f.write(
                _format_array_table(
                    pca_resultado, columns=pca_columns, index_prefix="Sample_"
                )
            )
            f.write("\n\n")
        else:
            f.write("6. PCA SCORES / PCA RESULT\n")
            f.write("-" * 70 + "\n")
            f.write("No PCA result available.\n\n")

        if tsne_resultado is not None:
            f.write("7. t-SNE RESULT\n")
            f.write("-" * 70 + "\n")
            tsne_array = np.asarray(tsne_resultado)
            tsne_columns = (
                [f"tSNE_{i+1}" for i in range(tsne_array.shape[1])]
                if tsne_array.ndim == 2
                else None
            )
            f.write(
                _format_array_table(
                    tsne_resultado, columns=tsne_columns, index_prefix="Sample_"
                )
            )
            f.write("\n\n")
        else:
            f.write("7. t-SNE RESULT\n")
            f.write("-" * 70 + "\n")
            f.write("No t-SNE result available.\n\n")

        if tsne_pca_resultado is not None:
            f.write("8. t-SNE(PCA(X)) RESULT\n")
            f.write("-" * 70 + "\n")
            tsne_pca_array = np.asarray(tsne_pca_resultado)
            tsne_pca_columns = (
                [f"tSNE_PCA_{i+1}" for i in range(tsne_pca_array.shape[1])]
                if tsne_pca_array.ndim == 2
                else None
            )
            f.write(
                _format_array_table(
                    tsne_pca_resultado, columns=tsne_pca_columns, index_prefix="Sample_"
                )
            )
            f.write("\n\n")
        else:
            f.write("8. t-SNE(PCA(X)) RESULT\n")
            f.write("-" * 70 + "\n")
            f.write("No t-SNE(PCA(X)) result available.\n\n")

        f.write("9. INTERPRETATION NOTES\n")
        f.write("-" * 70 + "\n")
        f.write(
            "PCA is an exploratory dimensionality reduction method. The explained "
            "variance indicates how much of the total data variability is represented "
            "by each principal component. A high explained variance does not necessarily "
            "mean that the component contains chemically relevant information; it may "
            "also reflect baseline effects, global intensity differences, noise, or "
            "instrumental variation.\n\n"
        )
        f.write(
            "Scores show the position of samples in the reduced PCA space. Loadings "
            "should be analyzed separately to identify which original variables or "
            "spectral regions contribute to the observed separation.\n\n"
        )
        f.write(
            "t-SNE is mainly useful for visualization and neighborhood structure. "
            "Its axes do not have the same direct variance interpretation as PCA axes.\n"
        )

        f.write(
            "Direct t-SNE dimensions: " f"{tsne_parameters.get('direct_dimensions')}\n"
        )
        f.write(
            "Direct t-SNE perplexity: " f"{tsne_parameters.get('direct_perplexity')}\n"
        )
        f.write(
            "Direct t-SNE iterations: " f"{tsne_parameters.get('direct_iterations')}\n"
        )
        f.write("PCA+t-SNE perplexity: " f"{tsne_parameters.get('pca_perplexity')}\n")
        f.write("PCA+t-SNE iterations: " f"{tsne_parameters.get('pca_iterations')}\n")

    print(f"Report generated: {report_path}")
    return str(report_path)