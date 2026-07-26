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
from plotting import calculate_accuracy

from core.translations import translate, get_language


def tr(text, **values):
    return translate(text, get_language(), **values)


def pca(X, componentes, return_model=False):
    """
    Perform Principal Component Analysis (PCA) on a data matrix and return the transformed data along with explained variance percentages. This is used to reduce dimensionality while retaining as much variability in the data as possible.

    The function validates the requested number of components against the data shape, fits a PCA model with that number of components, and returns both the projected data and the percentage of variance explained by each retained component. If the requested number of components is outside the allowed range, it raises a ValueError.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix where rows are samples and columns are variables or
        features to be analyzed.
    componentes : int or str
        Desired number of principal components. Must be an integer between 2 and
        the minimum of the number of samples and the number of variables.

    Returns
    -------
    tuple
        A tuple (dato_pca, varianza_porcentaje) where:
        - dato_pca is a NumPy array of shape (n_samples, componentes) containing
          the data projected onto the principal components.
        - varianza_porcentaje is a one-dimensional NumPy array with the
          percentage of explained variance for each principal component.

    Raises
    ------
    ValueError
        If the requested number of components is not greater than 1 or exceeds
        the allowable maximum based on the input data dimensions.
    """
    componentes = int(componentes)
    num_muestras, num_variables = X.shape
    max_pc = min(num_muestras, num_variables)

    if 1 < componentes <= max_pc:
        modelo_pca = PCA(n_components=componentes)
        dato_pca = modelo_pca.fit_transform(X)
        varianza_porcentaje = modelo_pca.explained_variance_ratio_ * 100
        if return_model:
            return dato_pca, varianza_porcentaje, modelo_pca
        return dato_pca, varianza_porcentaje
    else:
        raise ValueError(
            tr(
                "The number of components must be between 2 and {maximum}.",
                maximum=max_pc,
            )
        )


def plot_pca_2d(
    dato_pca,
    varianza_porcentaje,
    asignacion_colores,
    types,
    componentes_x,
    componentes_y,
    intervalo_confianza,
):
    """
    Create a 2D PCA scatter plot with optional confidence ellipses for each group. This visualization helps assess sample clustering and the variance captured by selected principal components.

    The function selects two principal component axes, builds a DataFrame with the projected scores and associated type labels, and computes an accuracy metric using a helper classifier-based function. It then constructs an interactive Plotly figure with colored points for each group and, when possible, draws confidence ellipses capturing the dispersion of each group.

    Parameters
    ----------
    dato_pca : array-like of shape (n_samples, n_components)
        Matrix of PCA-transformed data where rows are samples and columns are
        principal components.
    varianza_porcentaje : array-like of shape (n_components,)
        Percentage of explained variance for each principal component, expressed
        as values between 0 and 100.
    asignacion_colores : dict
        Mapping from type labels to color strings compatible with Plotly, used
        to color points and ellipses for each group.
    types : pandas.Series or array-like
        Sequence of type or class labels corresponding to each sample in
        `dato_pca`. It is reset to a simple RangeIndex internally if it is a
        Series.
    componentes_x : int or sequence of int
        Index or one-element sequence indicating which principal component to
        display on the x-axis, using 1-based indexing.
    componentes_y : int or sequence of int
        Index or one-element sequence indicating which principal component to
        display on the y-axis, using 1-based indexing.
    intervalo_confianza : float or str
        Confidence level for the group ellipses expressed as a percentage
        (e.g., 95 for 95% confidence), which is converted to a [0, 1] value for
        the chi-square distribution.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Plotly Figure object containing the PCA scatter plot with per-group
        points, optional confidence ellipses, and layout configured to show
        explained variance and accuracy information in the title.
    """
    types = types.reset_index(drop=True)
    idx_x = (
        componentes_x[0]
        if isinstance(componentes_x, (list, np.ndarray))
        else componentes_x
    )
    idx_y = (
        componentes_y[0]
        if isinstance(componentes_y, (list, np.ndarray))
        else componentes_y
    )

    porcentaje_varianza_x = varianza_porcentaje[idx_x - 1]
    porcentaje_varianza_y = varianza_porcentaje[idx_y - 1]

    eje_x = dato_pca[:, idx_x - 1]
    eje_y = dato_pca[:, idx_y - 1]

    dato_2d = np.column_stack((eje_x, eje_y))

    # df_pca = pd.DataFrame(dato_2d, columns=["PC1", "PC2"])
    # df_pca["Type"] = types

    # accuracy = calculate_accuracy(df_pca, df_pca["Type"])

    # DESPUÉS:
    df_pca = pd.DataFrame(dato_2d, columns=["PC1", "PC2"])
    df_pca["Type"] = types

    # Para accuracy usar todas las componentes disponibles
    df_todas = pd.DataFrame(
        dato_pca, columns=[f"PC{i+1}" for i in range(dato_pca.shape[1])]
    )
    df_todas["Type"] = types
    accuracy = calculate_accuracy(df_todas, df_todas["Type"])

    fig = go.Figure()
    intervalo = float(intervalo_confianza) / 100

    for tipo in np.unique(types):
        indices = df_pca["Type"] == tipo
        fig.add_trace(
            go.Scatter(
                x=df_pca.loc[indices, "PC1"],
                y=df_pca.loc[indices, "PC2"],
                mode="markers",
                marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
                name=f"{tipo}",
            )
        )

        datos_tipo = df_pca.loc[indices, ["PC1", "PC2"]].to_numpy()

        if datos_tipo.shape[0] > 2 and not np.allclose(datos_tipo.std(axis=0), 0):
            centro = np.mean(datos_tipo, axis=0)
            cov = np.cov(datos_tipo.T)
            elipse = generate_ellipse(
                centro,
                cov,
                color=asignacion_colores[tipo],
                intervalo_confianza=intervalo,
            )
            fig.add_trace(elipse)
        else:
            print(
                f"⚠️ Group '{tipo}' has insufficient data or zero variance for ellipse generation."
            )

        fig.update_layout(
            title=None,
            xaxis_title=f"PC{idx_x} ({porcentaje_varianza_x:.2f}%)",
            yaxis_title=f"PC{idx_y} ({porcentaje_varianza_y:.2f}%)",
            legend=dict(
                title=dict(text=tr("Type")),
                x=0.98,
                y=0.02,
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.82)",
                bordercolor="rgba(0,0,0,0.30)",
                borderwidth=1,
                font=dict(family="Arial", size=12, color="black"),
            ),
            annotations=[
                dict(
                    text=tr(
                        "KNN accuracy (5-fold CV, k=3): {accuracy:.2f}%",
                        accuracy=accuracy,
                    ),
                    x=0.99,
                    y=1.06,
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    yanchor="bottom",
                    showarrow=False,
                    font=dict(family="Arial", size=15, color="black"),
                    bgcolor="rgba(255,255,255,0.75)",
                    bordercolor="rgba(0,0,0,0.18)",
                    borderwidth=1,
                    borderpad=5,
                )
            ],
            margin=dict(l=75, r=35, b=75, t=85),
            width=1100,
            height=700,
            plot_bgcolor="rgba(235,240,248,1)",
            paper_bgcolor="white",
            uirevision="keep-layout",
        )
    return fig


def generate_ellipse(
    centro, cov, num_puntos=100, color="rgba(150,150,150,0.3)", intervalo_confianza=0.95
):
    """
    Generate a 2D confidence ellipse from a covariance matrix and center point. This is used to visualize the spread and orientation of grouped data in PCA or t-SNE plots.

    The function computes the ellipse radii from the eigenvalues of the covariance matrix and a chi-square quantile for the requested confidence level, then rotates and translates the ellipse to the specified center. If the covariance matrix is not valid for decomposition, it returns an empty Plotly scatter trace to avoid breaking the visualization pipeline.

    Parameters
    ----------
    centro : array-like of shape (2,)
        Coordinates of the ellipse center in the target 2D space.
    cov : array-like of shape (2, 2)
        Symmetric positive semi-definite covariance matrix describing the
        dispersion of the data around `centro`.
    num_puntos : int, optional
        Number of points used to discretize the ellipse contour.
    color : str, optional
        RGBA or CSS-compatible color string used to draw the ellipse line.
    intervalo_confianza : float, optional
        Confidence level expressed as a decimal between 0 and 1 used to compute
        the chi-square quantile for scaling the ellipse radii.

    Returns
    -------
    plotly.graph_objs._scatter.Scatter
        Plotly Scatter trace representing the ellipse contour; if an error
        occurs during generation, the trace contains no points and is still
        safe to add to a figure.
    """
    try:
        U, S, _ = np.linalg.svd(cov)
        radii = np.sqrt(chi2.ppf(intervalo_confianza, df=2) * S)

        theta = np.linspace(0, 2 * np.pi, num_puntos)
        x = np.cos(theta)
        y = np.sin(theta)

        elipse = np.array([x, y]).T @ np.diag(radii) @ U.T + centro

        return go.Scatter(
            x=elipse[:, 0],
            y=elipse[:, 1],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
        )
    except Exception as e:
        print(f"Error generating ellipse: {e}")
        return go.Scatter(x=[], y=[], mode="lines", showlegend=False)


def plot_pca_3d(
    dato_pca,
    varianza_porcentaje,
    asignacion_colores,
    types,
    componentes_x,
    componentes_y,
    componentes_z,
    intervalo_confianza,
):
    """
    Create a 3D PCA scatter plot with optional confidence ellipsoids for each group. This visualization helps explore clustering structure and variance captured by three selected principal components.

    The function selects three principal component axes, builds a DataFrame with the projected scores and associated type labels, and computes an accuracy metric using a helper classifier-based function on the non-missing samples. It then constructs an interactive Plotly 3D figure with colored points for each group and, when enough samples and variance are present, draws confidence ellipsoids that approximate the spatial dispersion of each group.

    Parameters
    ----------
    dato_pca : array-like of shape (n_samples, n_components)
        Matrix of PCA-transformed data where rows are samples and columns are
        principal components.
    varianza_porcentaje : array-like of shape (n_components,)
        Percentage of explained variance for each principal component, expressed
        as values between 0 and 100.
    asignacion_colores : dict
        Mapping from type labels to color strings compatible with Plotly, used
        to color markers and ellipsoids for each group.
    types : pandas.Series or array-like
        Sequence of type or class labels corresponding to each sample in
        `dato_pca`.
    componentes_x : int or sequence of int
        Index or one-element sequence indicating which principal component to
        display on the x-axis, using 1-based indexing.
    componentes_y : int or sequence of int
        Index or one-element sequence indicating which principal component to
        display on the y-axis, using 1-based indexing.
    componentes_z : int or sequence of int
        Index or one-element sequence indicating which principal component to
        display on the z-axis, using 1-based indexing.
    intervalo_confianza : float or str
        Confidence level for the group ellipsoids expressed as a percentage
        (e.g., 95 for 95% confidence), which is converted to a [0, 1] value for
        the chi-square distribution.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Plotly Figure object containing the 3D PCA scatter plot with per-group
        points, optional confidence ellipsoids, and layout configured to show
        explained variance and accuracy information in the title.
    """
    idx_x = (
        componentes_x[0]
        if isinstance(componentes_x, (list, np.ndarray))
        else componentes_x
    )
    idx_y = (
        componentes_y[0]
        if isinstance(componentes_y, (list, np.ndarray))
        else componentes_y
    )
    idx_z = (
        componentes_z[0]
        if isinstance(componentes_z, (list, np.ndarray))
        else componentes_z
    )

    porcentaje_varianza_x = varianza_porcentaje[idx_x - 1]
    porcentaje_varianza_y = varianza_porcentaje[idx_y - 1]
    porcentaje_varianza_z = varianza_porcentaje[idx_z - 1]

    eje_x = dato_pca[:, idx_x - 1]
    eje_y = dato_pca[:, idx_y - 1]
    eje_z = dato_pca[:, idx_z - 1]

    dato_3d = np.column_stack((eje_x, eje_y, eje_z))

    """
    df_pca = pd.DataFrame(dato_3d, columns=["PC1", "PC2", "PC3"])
    df_pca["Type"] = types

    df_pca_clean = df_pca.dropna()

    if df_pca_clean["Type"].isnull().any():
        print("There are NaN values in the 'Type' column.")
    else:
        accuracy = calculate_accuracy(df_pca_clean, df_pca_clean["Type"])
        print(f"----Accuracy Percentage (PCA 3D)= {accuracy:.2f}%") 
    """

    # DESPUÉS:
    df_pca = pd.DataFrame(dato_3d, columns=["PC1", "PC2", "PC3"])
    df_pca["Type"] = types

    df_todas = pd.DataFrame(
        dato_pca, columns=[f"PC{i+1}" for i in range(dato_pca.shape[1])]
    )
    df_todas["Type"] = types
    df_todas_clean = df_todas.dropna()

    accuracy = calculate_accuracy(df_todas_clean, df_todas_clean["Type"])

    fig = go.Figure()
    intervalo = float(intervalo_confianza) / 100

    for tipo in np.unique(types):
        indices = df_pca["Type"] == tipo
        fig.add_trace(
            go.Scatter3d(
                x=df_pca.loc[indices, "PC1"],
                y=df_pca.loc[indices, "PC2"],
                z=df_pca.loc[indices, "PC3"],
                mode="markers",
                marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
                name=f"{tipo}",
            )
        )

        datos_tipo = df_pca.loc[indices, ["PC1", "PC2", "PC3"]].to_numpy()
        if datos_tipo.shape[0] > 3:
            centro = np.mean(datos_tipo, axis=0)
            cov = np.cov(datos_tipo.T)
            elipsoide = generar_elipsoide(
                centro, cov, asignacion_colores[tipo], intervalo
            )
            fig.add_trace(elipsoide)

    fig.update_layout(
        title=None,
        legend=dict(
            title=dict(
                text="Type",
                font=dict(size=14, family="Arial", color="black"),
            ),
            x=0.98,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=12, family="Arial", color="black"),
            itemsizing="constant",
            bordercolor="rgba(0,0,0,0.30)",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.82)",
        ),
        annotations=[
            dict(
                text=tr(
                        "KNN accuracy (5-fold CV, k=3): {accuracy:.2f}%",
                        accuracy=accuracy,
                    ),
                x=0.99,
                y=0.99,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                showarrow=False,
                font=dict(family="Arial", size=15, color="black"),
                bgcolor="rgba(255,255,255,0.78)",
                bordercolor="rgba(0,0,0,0.18)",
                borderwidth=1,
                borderpad=5,
            )
        ],
        scene=dict(
            xaxis_title=f"PC{idx_x} ({porcentaje_varianza_x:.2f}%)",
            yaxis_title=f"PC{idx_y} ({porcentaje_varianza_y:.2f}%)",
            zaxis_title=f"PC{idx_z} ({porcentaje_varianza_z:.2f}%)",
            camera=dict(
                eye=dict(x=1.45, y=1.45, z=1.25),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        margin=dict(l=20, r=20, b=25, t=25),
        width=1100,
        height=760,
        uirevision="keep-legend-position",
    )

    return fig


def generar_elipsoide(centro, cov, color="rgba(150,150,150,0.3)", intervalo=0.95):
    """
    Generate a 3D confidence ellipsoid from a covariance matrix and center point. This helps visualize the spatial spread and orientation of grouped samples in three-dimensional PCA or t-SNE spaces.

    The function computes the ellipsoid radii from the covariance eigenvalues and a chi-square quantile for the requested confidence level, then rotates and translates a spherical grid so that it is centered at the given point and aligned with the covariance structure. It returns a Plotly Surface object that can be added to 3D scatter plots as a semi-transparent ellipsoid.

    Parameters
    ----------
    centro : array-like of shape (3,)
        Coordinates of the ellipsoid center in the target 3D space.
    cov : array-like of shape (3, 3)
        Symmetric positive semi-definite covariance matrix describing the
        dispersion of the data around `centro`.
    color : str, optional
        RGBA or CSS-compatible color string used to color the ellipsoid
        surface.
    intervalo : float, optional
        Confidence level expressed as a decimal between 0 and 1 used to compute
        the chi-square quantile for scaling the ellipsoid radii.

    Returns
    -------
    plotly.graph_objs._figure.Surface
        Plotly Surface trace representing the confidence ellipsoid, suitable
        for overlay on 3D scatter plots.
    """
    intervalo_confianza = intervalo

    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(
        chi2.ppf(intervalo_confianza, df=3) * S
    )  # 0.999 so that the ellipsoid encloses as many samples as possible

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = centro + np.dot(
                U, np.multiply(radii, [x[i, j], y[i, j], z[i, j]])
            )

    return go.Surface(
        x=x, y=y, z=z, opacity=0.3, colorscale=[[0, color], [1, color]], showscale=False
    )


def tsne(df, n_componentes, perplexity=30, learning_rate=200, max_iter=1000):
    """
    Apply t-SNE to reduce high-dimensional data to a 2D or 3D embedding. This function configures and validates core t-SNE parameters to produce a stable, interpretable low-dimensional representation.

    The function enforces valid output dimensionality, iteration count, and perplexity range based on the number of samples, then initializes a t-SNE model with PCA-based initialization and a fixed random seed. It returns the transformed coordinates, suitable for visualization or subsequent analysis.

    Parameters
    ----------
    df : pandas.DataFrame or array-like of shape (n_samples, n_features)
        Input data matrix where rows are samples and columns are features to be
        embedded.
    n_componentes : int
        Target dimensionality of the t-SNE embedding; must be either 2 or 3.
    perplexity : float, optional
        Initial perplexity value guiding the balance between local and global
        structure; internally clipped to [1, min(30, n_samples - 1)].
    learning_rate : float, optional
        Learning rate used by the t-SNE optimization algorithm.
    max_iter : int, optional
        Maximum number of optimization iterations; must be at least 250.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_samples, n_componentes) containing the t-SNE
        embedding coordinates.

    Raises
    ------
    ValueError
        If `n_componentes` is not 2 or 3, if `max_iter` is less than 250, if
        `perplexity` is not greater than 0, or if there are fewer than two
        samples in `df`.
    """
    componentes = n_componentes

    componentes = int(componentes)

    if componentes not in (2, 3):
        raise ValueError(tr("t-SNE output dimensions must be 2 or 3."))

    max_iter = int(max_iter)

    if max_iter < 250:
        raise ValueError(tr("t-SNE iterations must be at least 250."))

    perplexity = float(perplexity)

    if perplexity <= 0:
        raise ValueError(tr("t-SNE perplexity must be greater than 0."))

    n_samples = df.shape[0]

    if n_samples < 2:
        raise ValueError(tr("t-SNE requires at least two samples."))

    perplexity = min(
        perplexity,
        30.0,
        float(n_samples - 1),
    )

    perplexity = max(perplexity, 1.0)
    tsne = TSNE(
        n_components=componentes,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        init="pca",
        random_state=42,
    )
    datos_transformados = tsne.fit_transform(df)

    return datos_transformados


def plot_tsne_2d(dato_tsne, tipos, asignacion_colores, intervalo=0.95):
    """
    Create a 2D t-SNE scatter plot with optional confidence ellipses for each group. This visualization highlights local neighborhood structure in a low-dimensional embedding and summarizes classification performance with a k-NN based accuracy estimate.

    The function builds a DataFrame from the t-SNE coordinates and associated type labels, scales the valid points, and trains a k-NN classifier to compute an accuracy percentage on a held-out test split. It then constructs an interactive Plotly scatter plot colored by type and, for groups with enough points, overlays confidence ellipses derived from the empirical covariance of each group.

    Parameters
    ----------
    dato_tsne : array-like of shape (n_samples, 2)
        Two-dimensional t-SNE embedding where each row corresponds to one
        sample and the columns represent the t-SNE components.
    tipos : pandas.Series or array-like
        Sequence of type or class labels corresponding to each embedded sample.
        If a Series is provided, its index is reset before use.
    asignacion_colores : dict
        Mapping from type labels to color strings compatible with Plotly, used
        to color the scatter markers and ellipses for each group.
    intervalo : float, optional
        Confidence level expressed as a decimal between 0 and 1 used to compute
        the chi-square quantile that scales the ellipses around each group.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Plotly Figure object containing the 2D t-SNE scatter plot with per-
        group points, optional confidence ellipses, and a title summarizing the
        k-NN classification accuracy on the embedding.
    """

    if isinstance(tipos, pd.Series):
        tipos = tipos.reset_index(drop=True)

    df = pd.DataFrame(dato_tsne, columns=["X Axis", "Y Axis"])
    df["Type"] = tipos
    df["Color"] = [asignacion_colores[t] for t in tipos]

    df_clean = df.dropna()
    accuracy = calculate_accuracy(df_clean, df_clean["Type"])

    fig = px.scatter(
        df,
        x="X Axis",
        y="Y Axis",
        color="Type",
        color_discrete_map=asignacion_colores,
        hover_name="Type",
        labels={
            "Type": "Type",
        },
    )

    fig.update_layout(
        title=None,
        xaxis_title=tr("Component 1"),
        yaxis_title=tr("Component 2"),
        legend=dict(
            title=dict(text=tr("Type")),
            x=0.98,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(0,0,0,0.30)",
            borderwidth=1,
            font=dict(family="Arial", size=12, color="black"),
        ),
        annotations=[
            dict(
                text=tr(
                        "KNN accuracy (5-fold CV, k=3): {accuracy:.2f}%",
                        accuracy=accuracy,
                    ),
                x=0.99,
                y=1.06,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(family="Arial", size=15, color="black"),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="rgba(0,0,0,0.18)",
                borderwidth=1,
                borderpad=5,
            )
        ],
        margin=dict(l=75, r=35, b=75, t=85),
        width=1100,
        height=700,
        plot_bgcolor="rgba(235,240,248,1)",
        paper_bgcolor="white",
        uirevision="keep-layout",
    )

    fig.update_traces(marker=dict(size=6, opacity=0.85))

    for tipo in df["Type"].unique():
        grupo = df[df["Type"] == tipo][["X Axis", "Y Axis"]].values

        if grupo.shape[0] < 3:
            continue

        centro = grupo.mean(axis=0)
        cov = np.cov(grupo.T)

        valores, vectores = np.linalg.eigh(cov)
        orden = valores.argsort()[::-1]
        valores = valores[orden]
        vectores = vectores[:, orden]

        chi2_val = chi2.ppf(intervalo, df=2)
        angulos = np.linspace(0, 2 * np.pi, 100)

        elipse = np.array([np.cos(angulos), np.sin(angulos)])

        escala = np.diag(np.sqrt(valores * chi2_val))
        elipse_transf = vectores @ escala @ elipse + centro[:, None]

        fig.add_trace(
            go.Scatter(
                x=elipse_transf[0],
                y=elipse_transf[1],
                mode="lines",
                line=dict(color=asignacion_colores[tipo], dash="solid", width=2),
                name=f"Ellipse {tipo}",
                showlegend=False,
            )
        )

    return fig


def generar_elipsoide_tsne(centro, cov, color="rgba(150,150,150,0.3)", intervalo=0.95):
    """
    Generate a 3D confidence ellipsoid for t-SNE embeddings from a covariance matrix and center point. This is used to visualize the spatial dispersion and orientation of groups of samples in 3D t-SNE plots.

    The function derives ellipsoid radii from the covariance eigenvalues and a chi-square quantile for the requested confidence level, then deforms and rotates a spherical grid so that it matches the covariance structure and is centered at the given point. It returns a Plotly Surface trace that can be overlaid on 3D scatter plots as a semi-transparent ellipsoid.

    Parameters
    ----------
    centro : array-like of shape (3,)
        Coordinates of the ellipsoid center in the 3D t-SNE space.
    cov : array-like of shape (3, 3)
        Symmetric positive semi-definite covariance matrix describing the
        dispersion of the data around `centro`.
    color : str, optional
        RGBA or CSS-compatible color string used to color the ellipsoid
        surface.
    intervalo : float, optional
        Confidence level expressed as a decimal between 0 and 1 used to compute
        the chi-square quantile for scaling the ellipsoid radii.

    Returns
    -------
    plotly.graph_objs._figure.Surface
        Plotly Surface trace representing the confidence ellipsoid in 3D t-SNE
        space, suitable for overlay on 3D scatter plots.
    """
    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(chi2.ppf(intervalo, df=3) * S)

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = centro + np.dot(
                U, np.multiply(radii, [x[i, j], y[i, j], z[i, j]])
            )

    return go.Surface(
        x=x, y=y, z=z, opacity=0.3, colorscale=[[0, color], [1, color]], showscale=False
    )


def plot_tsne_3d(dato_tsne, tipos, asignacion_colores, intervalo=0.95):
    """
    Create a 3D t-SNE scatter plot with optional confidence ellipsoids for each group. This visualization shows how samples cluster in three dimensions and summarizes classification performance with a k-NN based accuracy metric.

    The function constructs a DataFrame from the 3D t-SNE coordinates and type labels, removes rows with missing data, and computes an accuracy score using a helper classifier function. It then builds an interactive Plotly 3D scatter plot colored by type and, for groups with enough samples, overlays confidence ellipsoids derived from their empirical covariance.

    Parameters
    ----------
    dato_tsne : array-like of shape (n_samples, 3)
        Three-dimensional t-SNE embedding where each row corresponds to one
        sample and the columns represent the t-SNE components.
    tipos : pandas.Series or array-like
        Sequence of type or class labels corresponding to each embedded sample;
        if a Series is provided, its index is reset before use.
    asignacion_colores : dict
        Mapping from type labels to color strings compatible with Plotly, used
        to color the scatter markers and ellipsoids for each group.
    intervalo : float, optional
        Confidence level expressed as a decimal between 0 and 1 used to compute
        the chi-square quantile for scaling the ellipsoids around each group.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Plotly Figure object containing the 3D t-SNE scatter plot with per-
        group points, optional confidence ellipsoids, and a title summarizing
        the k-NN classification accuracy on the embedding.
    """
    tipos = tipos.reset_index(drop=True)
    df = pd.DataFrame(dato_tsne, columns=["X Axis", "Y Axis", "Z Axis"])
    df["Type"] = tipos
    df["Color"] = [asignacion_colores[t] for t in tipos]

    """
    df_knn = df.copy()

    df_knn = df_knn.dropna()

    if df_knn["Type"].isnull().any():
        print("There are NaN values in the 'Type' column.")
    else:
        accuracy = calculate_accuracy(df_knn, df_knn["Type"])
    """

    # DESPUÉS:
    df_knn = df.dropna()
    accuracy = calculate_accuracy(df_knn, df_knn["Type"])

    fig = go.Figure()

    for tipo in df["Type"].unique():
        grupo = df[df["Type"] == tipo][["X Axis", "Y Axis", "Z Axis"]].values

        fig.add_trace(
            go.Scatter3d(
                x=grupo[:, 0],
                y=grupo[:, 1],
                z=grupo[:, 2],
                mode="markers",
                marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
                name=f"{tipo}",
            )
        )

        if grupo.shape[0] >= 4:
            centro = grupo.mean(axis=0)
            cov = np.cov(grupo.T)
            elipsoide = generar_elipsoide_tsne(
                centro, cov, asignacion_colores[tipo], intervalo
            )
            fig.add_trace(elipsoide)

        fig.update_layout(
            title=None,
            legend=dict(
                title=dict(
                    text="Type",
                    font=dict(size=14, family="Arial", color="black"),
                ),
                x=0.98,
                y=0.02,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=12, family="Arial", color="black"),
                itemsizing="constant",
                bordercolor="rgba(0,0,0,0.30)",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.82)",
            ),
            annotations=[
                dict(
                    text=tr(
                        "KNN accuracy (5-fold CV, k=3): {accuracy:.2f}%",
                        accuracy=accuracy,
                    ),
                    x=0.99,
                    y=0.99,
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    yanchor="top",
                    showarrow=False,
                    font=dict(family="Arial", size=15, color="black"),
                    bgcolor="rgba(255,255,255,0.78)",
                    bordercolor="rgba(0,0,0,0.18)",
                    borderwidth=1,
                    borderpad=5,
                )
            ],
            scene=dict(
                xaxis_title=tr("Component 1"),
                yaxis_title=tr("Component 2"),
                zaxis_title="Component 3",
                camera=dict(
                    eye=dict(x=1.45, y=1.45, z=1.25),
                    center=dict(x=0, y=0, z=0),
                ),
            ),
            margin=dict(l=20, r=20, b=25, t=25),
            width=1100,
            height=760,
            uirevision="keep-legend-position",
        )

    return fig


def tsne_pca(df, cp_pca, cp_tsne, perplexity=30, learning_rate=200, max_iter=1000):
    """
    Apply PCA followed by t-SNE to obtain a low-dimensional embedding. This two-stage pipeline first reduces noise and redundancy with PCA and then preserves local neighborhood structure with t-SNE.

    The function runs PCA on the input data using the specified number of components and feeds the PCA scores into t-SNE with its own target dimensionality and optimization parameters. It returns the final t-SNE coordinates, enabling visualization or further analysis in a compact space.

    Parameters
    ----------
    df : pandas.DataFrame or array-like of shape (n_samples, n_features)
        High-dimensional input data where rows are samples and columns are
        features to be reduced.
    cp_pca : int
        Number of principal components to retain in the initial PCA step.
    cp_tsne : int
        Number of t-SNE components (target dimensions) to compute from the PCA
        scores.
    perplexity : float, optional
        Initial perplexity hint passed to the t-SNE step; the t-SNE function
        may internally adjust this based on sample size.
    learning_rate : float, optional
        Learning rate used by the t-SNE optimization algorithm.
    max_iter : int, optional
        Maximum number of iterations for the t-SNE optimization.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_samples, cp_tsne) containing the t-SNE embeddings
        obtained from the PCA-transformed data.
    """
    dato_pca, _ = pca(df, cp_pca)

    tsne_resultado = tsne(
        dato_pca,
        n_componentes=cp_tsne,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
    )

    return tsne_resultado