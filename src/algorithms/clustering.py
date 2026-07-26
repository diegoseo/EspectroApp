"""Hierarchical clustering algorithms used by EspectroApp."""

from collections import Counter, defaultdict

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist, squareform

from core.translations import translate, get_language


def tr(text, **values):
    return translate(text, get_language(), **values)


def _composition_text(labels):
    """Return a compact composition summary preserving first-seen order."""
    clean_labels = [
        tr("Unknown") if pd.isna(label) else str(label).strip() for label in labels
    ]

    counts = Counter(clean_labels)
    order = list(dict.fromkeys(clean_labels))

    return ", ".join(f"{counts[label]} {label}" for label in order)


def calculate_hca(dato, raman_shift, opciones, muestras_hca):
    """
    Perform HCA and return both the dendrogram and a cluster summary table.

    Returns
    -------
    tuple
        ``(figure, cluster_table)`` where ``cluster_table`` contains the
        columns Cluster, Label, Size and Composition.
    """
    del raman_shift  # Kept in the signature for backward compatibility.

    dato = dato.dropna()
    dato = dato.apply(pd.to_numeric, errors="coerce").dropna().astype(float)

    datos = dato.iloc[:, 1:]

    if datos.shape[1] < 2:
        raise ValueError(tr("HCA requires at least two valid samples."))

    claves = list(opciones.keys())
    metodo_distancia = claves[0] if len(claves) > 0 else None
    metodo_enlace = claves[1] if len(claves) > 1 else None

    if metodo_distancia == "Euclidiana":
        nombre_plot = "Euclidean"
        distancia = pdist(
            datos.T,
            metric="euclidean",
        )

    elif metodo_distancia == "Manhattan":
        nombre_plot = "Manhattan"
        distancia = pdist(
            datos.T,
            metric="cityblock",
        )

    elif metodo_distancia == "Coseno":
        nombre_plot = "Cosine"
        distancia = pdist(
            datos.T,
            metric="cosine",
        )

    elif metodo_distancia == "Chebyshev":
        nombre_plot = "Chebyshev"
        distancia = pdist(
            datos.T,
            metric="chebyshev",
        )

    elif metodo_distancia in {
        "Pearson",
        "Correlación Pearson",
        "Pearson correlation",
    }:
        nombre_plot = "Pearson"
        correlacion = datos.corr(method="pearson")
        distancia = squareform(
            1 - correlacion,
            checks=False,
        )

    elif metodo_distancia in {
        "Spearman",
        "Correlación Spearman",
        "Spearman correlation",
    }:
        nombre_plot = "Spearman"
        correlacion = datos.corr(method="spearman")
        distancia = squareform(
            1 - correlacion,
            checks=False,
        )

    elif metodo_distancia == "Jaccard":
        nombre_plot = "Jaccard"
        distancia = pdist(
            datos.T,
            metric="jaccard",
        )

    else:
        raise ValueError(tr("Unrecognized distance method"))

    if not np.isfinite(distancia).all():
        raise ValueError(
            tr(
                "The selected distance metric produced non-finite values. "
                "Check constant or invalid spectra."
            )
        )

    if metodo_enlace == "Ward":
        nombre_enlace = "ward"
        dendrograma = sch.linkage(
            distancia,
            method="ward",
        )

    elif metodo_enlace == "Single Linkage":
        nombre_enlace = "single"
        dendrograma = sch.linkage(
            distancia,
            method="single",
        )

    elif metodo_enlace == "Complete Linkage":
        nombre_enlace = "complete"
        dendrograma = sch.linkage(
            distancia,
            method="complete",
        )

    elif metodo_enlace == "Average Linkage":
        nombre_enlace = "average"
        dendrograma = sch.linkage(
            distancia,
            method="average",
        )

    else:
        raise ValueError(tr("Unrecognized linkage method"))

    p = int(
        opciones.get(
            "Numero Clusters",
            12,
        )
    )

    p = max(
        2,
        min(
            p,
            datos.shape[1],
        ),
    )

    grupos = fcluster(
        dendrograma,
        t=p,
        criterion="maxclust",
    )

    muestras_por_grupo = defaultdict(list)

    for idx, grupo_id in enumerate(grupos):
        muestras_por_grupo[int(grupo_id)].append(idx)

    ddata_full = sch.dendrogram(
        dendrograma,
        no_plot=True,
    )

    orden_hojas = ddata_full["leaves"]

    grupos_ordenados = sorted(
        muestras_por_grupo,
        key=lambda gid: min(orden_hojas.index(i) for i in muestras_por_grupo[gid]),
    )

    etiquetas_nuevas = [f"C{len(muestras_por_grupo[gid])}" for gid in grupos_ordenados]

    # Create the figure without using pyplot.
    fig = Figure(figsize=(16, 8))

    # Explicitly create the axis that will be embedded in Qt.
    ax = fig.add_subplot(111)

    # Draw the dendrogram directly on the figure axis.
    sch.dendrogram(
        dendrograma,
        truncate_mode="lastp",
        p=p,
        leaf_rotation=90,
        show_leaf_counts=False,
        no_labels=True,
        ax=ax,
    )

    posiciones = np.arange(
        5,
        10 * len(etiquetas_nuevas),
        10,
    )

    ax.set_xticks(posiciones)

    ax.set_xticklabels(
        etiquetas_nuevas,
        rotation=90,
    )

    ax.set_title(
        f"Dendrogram using {nombre_enlace} linkage with "
        f"{nombre_plot} distance (HCA)"
    )

    ax.set_xlabel(tr("Samples"))

    ax.set_ylabel(tr("Distance"))

    fig.tight_layout()

    rows = []

    for display_cluster, grupo_id in enumerate(
        grupos_ordenados,
        start=1,
    ):
        indices = muestras_por_grupo[grupo_id]

        labels = [muestras_hca[i] for i in indices]

        size = len(indices)

        rows.append(
            {
                "Cluster": display_cluster,
                "Label": f"C{size}",
                "Size": size,
                "Composition": _composition_text(labels),
            }
        )

    cluster_table = pd.DataFrame(
        rows,
        columns=[
            "Cluster",
            "Label",
            "Size",
            "Composition",
        ],
    )

    return fig, cluster_table