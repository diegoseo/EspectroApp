"""Canonical metadata for comparing spectral preprocessing histories."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_pipeline(options: dict[str, Any] | None) -> dict[str, Any]:
    """Return a stable, JSON-serializable preprocessing definition."""
    return _json_ready(dict(options or {}))


def pipeline_signature(options: dict[str, Any] | None) -> str:
    """Return a short deterministic fingerprint for preprocessing options."""
    canonical = canonical_pipeline(options)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_pipeline_metadata(dataframe) -> dict[str, Any]:
    """Read normalized preprocessing metadata from a DataFrame.

    Datasets without metadata are treated as raw/unprocessed datasets.
    """
    attrs = dict(getattr(dataframe, "attrs", {}) or {})
    options = canonical_pipeline(attrs.get("preprocessing_pipeline"))
    signature = str(attrs.get("preprocessing_signature") or pipeline_signature(options))
    return {
        "name": str(attrs.get("preprocessing_pipeline_name") or ""),
        "options": options,
        "signature": signature,
        "is_raw": not bool(options),
    }


def describe_pipeline(metadata: dict[str, Any] | None) -> str:
    """Create a compact technical description for UI details."""
    metadata = dict(metadata or {})
    options = dict(metadata.get("options") or {})
    name = str(metadata.get("name") or "").strip()
    if not options:
        return "Raw / no preprocessing"
    if name:
        return name
    return ", ".join(str(key) for key in options) or "Custom preprocessing"
