"""Built-in method catalogue for EspectroApp."""

from __future__ import annotations

from methods.registry import MethodDefinition, MethodRegistry, register_many


def create_default_method_registry() -> MethodRegistry:
    registry = MethodRegistry()
    register_many(
        registry,
        (
            MethodDefinition(
                method_id="preprocessing",
                name="Spectral preprocessing",
                category="preprocessing",
                description="Applies one or more preprocessing steps to spectra.",
                produces_dataset=True,
                aliases=("normalization", "smoothing", "derivative", "baseline correction"),
            ),
            MethodDefinition(
                method_id="pca",
                name="Principal component analysis",
                category="exploration",
                description="Reduces dimensionality using principal components.",
                produces_model=True,
                produces_figure=True,
                aliases=("pca analysis", "principal components", "loading"),
            ),
            MethodDefinition(
                method_id="tsne",
                name="t-SNE",
                category="exploration",
                description="Creates a nonlinear low-dimensional embedding.",
                produces_figure=True,
                aliases=("t-sne analysis", "tsne analysis"),
            ),
            MethodDefinition(
                method_id="hca",
                name="Hierarchical cluster analysis",
                category="clustering",
                description="Groups samples and creates a dendrogram.",
                produces_figure=True,
                aliases=("hierarchical clustering", "dendrogram"),
            ),
            MethodDefinition(
                method_id="data_fusion",
                name="Data fusion",
                category="fusion",
                description="Combines information from two or more datasets.",
                produces_dataset=True,
                aliases=("low-level fusion", "mid-level fusion"),
            ),
            MethodDefinition(
                method_id="knn",
                name="K-nearest neighbors",
                category="classification",
                description="Supervised classification using neighboring samples.",
                produces_model=True,
                aliases=("knn classification", "k-nearest"),
            ),
            MethodDefinition(
                method_id="svm",
                name="Support vector machine",
                category="classification",
                description="Supervised classification or regression using support vectors.",
                produces_model=True,
                aliases=("support vector", "svc", "svr"),
            ),
            MethodDefinition(
                method_id="random_forest",
                name="Random forest",
                category="classification",
                description="Ensemble model based on decision trees.",
                produces_model=True,
                aliases=("random forest classification",),
            ),
            MethodDefinition(
                method_id="pls",
                name="Partial least squares",
                category="regression",
                description="Latent-variable regression model.",
                produces_model=True,
                aliases=("pls regression", "pls-da"),
            ),
        ),
    )
    return registry
