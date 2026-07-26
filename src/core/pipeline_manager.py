"""Portable storage for reusable EspectroApp preprocessing pipelines."""

from __future__ import annotations

import json
import re

from datetime import datetime
from pathlib import Path
from typing import Any

from core.analysis_history import get_portable_data_directory
from core.translations import translate, get_language


def tr(text, **values):
    return translate(text, get_language(), **values)


class PipelineManager:
    """Save, load, list and delete preprocessing pipelines."""

    def __init__(self, directory: Path | None = None):
        self.directory = (
            Path(directory)
            if directory is not None
            else get_portable_data_directory() / "pipelines"
        )
        self.directory.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, name: str) -> str:
        cleaned = re.sub(
            r'[<>:"/\\|?*\x00-\x1f]+',
            "_",
            str(name).strip(),
        )
        cleaned = cleaned.strip(" ._")

        if not cleaned:
            raise ValueError(tr("The pipeline name is invalid."))

        return cleaned

    def _path_for(self, name: str) -> Path:
        return self.directory / f"{self._safe_filename(name)}.json"

    def save(
        self,
        name: str,
        options: dict[str, Any],
    ) -> str:
        safe_name = self._safe_filename(name)
        path = self._path_for(safe_name)

        payload = {
            "format_version": 1,
            "name": safe_name,
            "created_at": datetime.now().isoformat(),
            "pipeline_type": "spectral_preprocessing",
            "options": options,
        }

        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(path)

        return safe_name

    def load(self, name: str) -> dict[str, Any]:
        path = self._path_for(name)

        if not path.exists():
            raise FileNotFoundError(
                tr("Pipeline not found: {name}", name=name)
            )

        payload = json.loads(path.read_text(encoding="utf-8"))

        if payload.get("pipeline_type") != "spectral_preprocessing":
            raise ValueError(tr("The selected file is not a preprocessing pipeline."))

        options = payload.get("options")
        if not isinstance(options, dict):
            raise ValueError(tr("The pipeline options are invalid."))

        return payload

    def list_names(self) -> list[str]:
        names = []

        for path in self.directory.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                name = str(payload.get("name", path.stem)).strip()
                if name:
                    names.append(name)
            except (
                OSError,
                json.JSONDecodeError,
                TypeError,
            ):
                continue

        return sorted(
            set(names),
            key=str.casefold,
        )

    def delete(self, name: str) -> None:
        path = self._path_for(name)
        path.unlink(missing_ok=True)