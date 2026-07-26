"""Reusable execution of spectral preprocessing pipelines.

This module applies the same canonical pipeline used by the preprocessing UI,
without requiring a visible window or a background thread. It is used when a
saved PCA model needs to prepare new raw spectra exactly as its training data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.preprocessing_signature import canonical_pipeline, pipeline_signature
from functions import (
    calculate_first_derivative,
    calculate_second_derivative,
    correct_linear_baseline,
    correct_shirley_baseline,
    normalize_by_area,
    normalize_by_mean,
    smooth_gaussian_filter,
    smooth_moving_average,
    smooth_savitzky_golay,
)


def apply_preprocessing_pipeline(
    dataframe: pd.DataFrame,
    options: dict[str, Any] | None,
    *,
    pipeline_name: str = "",
) -> pd.DataFrame:
    """Apply a saved preprocessing pipeline to an internal-format dataset.

    Parameters
    ----------
    dataframe:
        EspectroApp internal-format DataFrame. Row 0 stores labels; column 0
        stores the spectral axis.
    options:
        Canonical pipeline options produced by the preprocessing page.
    pipeline_name:
        Optional visible name stored in the output metadata.
    """
    if dataframe is None or dataframe.empty or dataframe.shape[0] < 2 or dataframe.shape[1] < 2:
        raise ValueError("The dataset does not contain enough spectral data.")

    pipeline = canonical_pipeline(options)
    source = dataframe.copy(deep=True)
    header = source.iloc[0].copy()
    numeric = source.iloc[1:].copy()

    x_axis = pd.to_numeric(numeric.iloc[:, 0], errors="coerce")
    if x_axis.isna().any():
        raise ValueError("The spectral axis contains non-numeric or missing values.")

    intensities = numeric.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if intensities.isna().any().any():
        raise ValueError("The intensity matrix contains non-numeric or missing values.")

    linear = pipeline.get("correccion_lineal")
    if linear:
        intensities = correct_linear_baseline(
            intensities,
            x_axis,
            x_start=linear["x_start"],
            x_end=linear["x_end"],
        )

    shirley = pipeline.get("correccion_shirley")
    if shirley:
        intensities = correct_shirley_baseline(
            intensities,
            x_axis,
            x_start=shirley["x_start"],
            x_end=shirley["x_end"],
            tolerance=shirley.get("tolerance", 1e-6),
            max_iterations=shirley.get("max_iterations", 100),
        )

    mean_normalization = pipeline.get("normalizar_media") or {}
    if mean_normalization.get("activar"):
        intensities = normalize_by_mean(
            intensities,
            mean_normalization.get("metodo", "0-1"),
        )

    if pipeline.get("normalizar_area"):
        intensities = normalize_by_area(intensities, x_axis)

    savgol = pipeline.get("suavizar_sg")
    if savgol:
        intensities = smooth_savitzky_golay(
            intensities,
            int(savgol["ventana"]),
            int(savgol["orden"]),
        )

    gaussian = pipeline.get("suavizar_fg")
    if gaussian:
        intensities = smooth_gaussian_filter(
            intensities,
            float(gaussian["sigma"]),
        )

    moving = pipeline.get("suavizar_mm")
    if moving:
        intensities = smooth_moving_average(
            intensities,
            int(moving["ventana"]),
        )

    if pipeline.get("derivada_1"):
        intensities = calculate_first_derivative(intensities, x_axis)
    elif pipeline.get("derivada_2"):
        intensities = calculate_second_derivative(intensities, x_axis)

    axis_name = str(header.iloc[0]).strip()
    result = pd.concat(
        [x_axis.reset_index(drop=True), intensities.reset_index(drop=True)],
        axis=1,
    )
    result.columns = [axis_name] + list(header.iloc[1:])
    result = pd.concat(
        [pd.DataFrame([result.columns], columns=result.columns), result],
        ignore_index=True,
    )
    result.columns = [0] + list(range(1, result.shape[1]))

    attrs = dict(getattr(dataframe, "attrs", {}) or {})
    history = list(attrs.get("preprocessing_history", []))
    history.append(pipeline)
    attrs["preprocessing_history"] = history
    attrs["preprocessing_pipeline"] = pipeline
    attrs["preprocessing_signature"] = pipeline_signature(pipeline)
    attrs["preprocessing_pipeline_name"] = str(pipeline_name or "")
    attrs["source_dataset_id"] = str(attrs.get("dataset_id", ""))
    result.attrs = attrs
    return result
