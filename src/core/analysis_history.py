"""Persistent traceability for EspectroApp analysis operations."""

from __future__ import annotations

import csv
import json
import os
import sys

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from core.translations import translate, get_language


def tr(text, **values):
    return translate(text, get_language(), **values)


def get_portable_data_directory() -> Path:
    """
    Return the preferred EspectroApp data directory.

    In a PyInstaller executable, data are stored beside the .exe inside
    ``EspectroApp_data`` so they can travel with a portable copy.

    During development, data are stored beside the project folder.
    If the preferred directory is not writable, the function falls back
    to the current user's local application-data directory.
    """
    if getattr(sys, "frozen", False):
        base_directory = Path(sys.executable).resolve().parent
    else:
        # src/core/analysis_history.py -> project root
        base_directory = Path(__file__).resolve().parents[2]

    preferred_directory = base_directory / "EspectroApp_data"

    try:
        preferred_directory.mkdir(parents=True, exist_ok=True)

        test_file = preferred_directory / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)

        return preferred_directory

    except OSError:
        local_app_data = os.getenv("LOCALAPPDATA")

        if local_app_data:
            fallback_directory = Path(local_app_data) / "EspectroApp"
        else:
            fallback_directory = Path.home() / ".espectroapp"

        fallback_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return fallback_directory


@dataclass(frozen=True)
class AnalysisHistoryEntry:
    """One operation performed on a dataset."""

    dataset: str
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    output_dataset: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    source_datasets: tuple[str, ...] = field(default_factory=tuple)

    @property
    def timestamp_text(self) -> str:
        return self.timestamp.strftime("%d/%m/%Y  %H:%M:%S")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["source_datasets"] = list(self.source_datasets)
        return data

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "AnalysisHistoryEntry":
        timestamp_value = data.get("timestamp")

        try:
            timestamp = datetime.fromisoformat(str(timestamp_value))
        except (TypeError, ValueError):
            timestamp = datetime.now()

        return cls(
            dataset=str(data.get("dataset", tr("Unnamed dataset"))),
            operation=str(data.get("operation", tr("Unknown operation"))),
            timestamp=timestamp,
            output_dataset=(
                str(data["output_dataset"]) if data.get("output_dataset") else None
            ),
            parameters=dict(data.get("parameters") or {}),
            source_datasets=tuple(
                str(item) for item in (data.get("source_datasets") or ())
            ),
        )


class AnalysisHistoryManager(QObject):
    """Store, persist and export EspectroApp analysis history."""

    changed = Signal()

    def __init__(
        self,
        parent: QObject | None = None,
        storage_path: Path | None = None,
    ):
        super().__init__(parent)

        self.storage_path = (
            Path(storage_path)
            if storage_path is not None
            else (get_portable_data_directory() / "analysis_history.json")
        )

        self._entries: list[AnalysisHistoryEntry] = []

        self.load()

    @property
    def entries(
        self,
    ) -> tuple[AnalysisHistoryEntry, ...]:
        return tuple(self._entries)

    def add(
        self,
        dataset: str,
        operation: str,
        output_dataset: str | None = None,
        parameters: dict[str, Any] | None = None,
        source_datasets: list[str] | tuple[str, ...] | None = None,
    ) -> AnalysisHistoryEntry:
        dataset_name = str(dataset).strip() or tr("Unnamed dataset")

        operation_name = str(operation).strip()

        if not operation_name:
            raise ValueError(tr("The history operation cannot be empty."))

        entry = AnalysisHistoryEntry(
            dataset=dataset_name,
            operation=operation_name,
            output_dataset=(str(output_dataset).strip() if output_dataset else None),
            parameters=dict(parameters or {}),
            source_datasets=tuple(
                str(item).strip()
                for item in (source_datasets or ())
                if str(item).strip()
            ),
        )

        self._entries.append(entry)
        self.save()
        self.changed.emit()
        return entry

    def clear(self) -> None:
        if not self._entries:
            return

        self._entries.clear()
        self.save()
        self.changed.emit()

    def grouped_by_dataset(
        self,
    ) -> dict[
        str,
        list[AnalysisHistoryEntry],
    ]:
        grouped: dict[
            str,
            list[AnalysisHistoryEntry],
        ] = {}

        for entry in self._entries:
            grouped.setdefault(
                entry.dataset,
                [],
            ).append(entry)

        return grouped

    def save(self) -> None:
        """Save the complete history as JSON."""
        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = self.storage_path.with_suffix(".tmp")

        payload = {
            "format_version": 1,
            "entries": [entry.to_dict() for entry in self._entries],
        }

        temporary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(self.storage_path)

    def load(self) -> None:
        """Load an existing persistent history."""
        if not self.storage_path.exists():
            return

        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))

            entries_data = payload.get(
                "entries",
                [],
            )

            self._entries = [
                AnalysisHistoryEntry.from_dict(item)
                for item in entries_data
                if isinstance(item, dict)
            ]

        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
        ):
            # Keep the application usable even if
            # the history file is damaged.
            self._entries = []

    def export_json(self, path: str | Path) -> None:
        """Export history to a user-selected JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "format_version": 1,
            "exported_at": datetime.now().isoformat(),
            "entries": [entry.to_dict() for entry in self._entries],
        }

        output_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def export_csv(self, path: str | Path) -> None:
        """Export history to a flat CSV table."""
        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "timestamp",
                    "dataset",
                    "operation",
                    "output_dataset",
                    "source_datasets",
                    "parameters",
                ],
            )

            writer.writeheader()

            for entry in self._entries:
                writer.writerow(
                    {
                        "timestamp": (entry.timestamp.isoformat()),
                        "dataset": entry.dataset,
                        "operation": entry.operation,
                        "output_dataset": (entry.output_dataset or ""),
                        "source_datasets": " | ".join(entry.source_datasets),
                        "parameters": json.dumps(
                            entry.parameters,
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )