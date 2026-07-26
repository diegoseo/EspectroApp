"""Save and restore complete EspectroApp work sessions."""

from __future__ import annotations

import json
import pickle
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_EXTENSION = ".espectroapp"
PROJECT_FORMAT_VERSION = 3


class ProjectFormatError(ValueError):
    """Raised when a project file is missing required content or is incompatible."""


def normalize_project_path(path: str | Path) -> Path:
    output = Path(path)
    if output.suffix.lower() != PROJECT_EXTENSION:
        output = output.with_suffix(PROJECT_EXTENSION)
    return output


def save_project_file(
    path: str | Path,
    *,
    dataframes: list[Any],
    dataset_names: list[str],
    history_entries: list[dict[str, Any]],
    language: str,
    active_page: str,
    project_name: str,
    fitted_models: list[dict[str, Any]] | None = None,
    fitted_model_artifacts: dict[str, Any] | None = None,
) -> Path:
    """Write the current application session to one portable project archive."""
    destination = normalize_project_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if len(dataframes) != len(dataset_names):
        raise ValueError("Dataset names and matrices are not aligned.")

    metadata = {
        "format_version": PROJECT_FORMAT_VERSION,
        "application": "EspectroApp",
        "project_name": str(project_name).strip() or destination.stem,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "language": language,
        "active_page": active_page,
        "dataset_names": [str(name) for name in dataset_names],
        "dataset_count": len(dataframes),
        "history_entries": history_entries,
        "fitted_models": list(fitted_models or []),
    }

    with tempfile.TemporaryDirectory(prefix="espectroapp_project_") as temp_dir:
        root = Path(temp_dir)
        (root / "project.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        with (root / "datasets.pkl").open("wb") as stream:
            pickle.dump(dataframes, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with (root / "model_artifacts.pkl").open("wb") as stream:
            pickle.dump(dict(fitted_model_artifacts or {}), stream, protocol=pickle.HIGHEST_PROTOCOL)

        temporary_destination = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with zipfile.ZipFile(
                temporary_destination,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                archive.write(root / "project.json", "project.json")
                archive.write(root / "datasets.pkl", "datasets.pkl")
                archive.write(root / "model_artifacts.pkl", "model_artifacts.pkl")
            temporary_destination.replace(destination)
        finally:
            temporary_destination.unlink(missing_ok=True)

    return destination


def load_project_file(path: str | Path) -> dict[str, Any]:
    """Read and validate an EspectroApp project archive."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    try:
        with zipfile.ZipFile(source, "r") as archive:
            names = set(archive.namelist())
            required = {"project.json", "datasets.pkl"}
            missing = required - names
            if missing:
                raise ProjectFormatError(
                    "The project is incomplete. Missing: " + ", ".join(sorted(missing))
                )

            metadata = json.loads(archive.read("project.json").decode("utf-8"))
            dataframes = pickle.loads(archive.read("datasets.pkl"))
            if "model_artifacts.pkl" in names:
                fitted_model_artifacts = pickle.loads(archive.read("model_artifacts.pkl"))
            else:
                fitted_model_artifacts = {}
    except zipfile.BadZipFile as error:
        raise ProjectFormatError("The selected file is not a valid EspectroApp project.") from error
    except (json.JSONDecodeError, pickle.UnpicklingError, EOFError) as error:
        raise ProjectFormatError("The project content is damaged or unreadable.") from error

    version = int(metadata.get("format_version", 0))
    if version > PROJECT_FORMAT_VERSION:
        raise ProjectFormatError(
            "This project was created with a newer EspectroApp project format."
        )
    if not isinstance(dataframes, list):
        raise ProjectFormatError("The project dataset collection is invalid.")

    dataset_names = metadata.get("dataset_names", [])
    if not isinstance(dataset_names, list) or len(dataset_names) != len(dataframes):
        raise ProjectFormatError("The project dataset names do not match its matrices.")

    return {
        "project_name": str(metadata.get("project_name") or source.stem),
        "language": str(metadata.get("language") or "en"),
        "active_page": str(metadata.get("active_page") or "welcome"),
        "dataset_names": [str(name) for name in dataset_names],
        "dataframes": dataframes,
        "history_entries": list(metadata.get("history_entries") or []),
        "fitted_models": list(metadata.get("fitted_models") or []),
        "fitted_model_artifacts": dict(fitted_model_artifacts or {}),
    }
