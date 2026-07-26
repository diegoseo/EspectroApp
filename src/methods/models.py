"""Records for fitted and reusable models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any
from uuid import uuid4

try:
    from PySide6.QtCore import QObject, Signal
except ImportError:  # Allows core registry tests without the optional GUI stack.
    class _BoundSignal:
        def __init__(self) -> None:
            self._callbacks = []

        def connect(self, callback) -> None:
            self._callbacks.append(callback)

        def emit(self) -> None:
            for callback in tuple(self._callbacks):
                callback()

    class Signal:
        def __set_name__(self, owner, name) -> None:
            self.storage_name = f"__signal_{name}"

        def __get__(self, instance, owner):
            if instance is None:
                return self
            signal = instance.__dict__.get(self.storage_name)
            if signal is None:
                signal = _BoundSignal()
                instance.__dict__[self.storage_name] = signal
            return signal

    class QObject:
        def __init__(self, parent=None) -> None:
            self.parent = parent


@dataclass(frozen=True)
class FittedModelRecord:
    model_id: str
    method_id: str
    name: str
    dataset: str
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(
        cls,
        *,
        method_id: str,
        name: str,
        dataset: str,
        parameters: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        artifact_path: str | None = None,
    ) -> "FittedModelRecord":
        return cls(
            model_id=str(uuid4()),
            method_id=str(method_id).strip().lower(),
            name=str(name).strip() or str(method_id),
            dataset=str(dataset).strip(),
            parameters=dict(parameters or {}),
            metrics=dict(metrics or {}),
            artifact_path=artifact_path,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FittedModelRecord":
        try:
            created_at = datetime.fromisoformat(str(payload.get("created_at")))
        except (TypeError, ValueError):
            created_at = datetime.now()
        return cls(
            model_id=str(payload.get("model_id") or uuid4()),
            method_id=str(payload.get("method_id") or "unknown"),
            name=str(payload.get("name") or payload.get("method_id") or "Model"),
            dataset=str(payload.get("dataset") or ""),
            parameters=dict(payload.get("parameters") or {}),
            metrics=dict(payload.get("metrics") or {}),
            artifact_path=(str(payload["artifact_path"]) if payload.get("artifact_path") else None),
            created_at=created_at,
        )


class FittedModelManager(QObject):
    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._records: list[FittedModelRecord] = []
        self._artifacts: dict[str, Any] = {}

    @property
    def records(self) -> tuple[FittedModelRecord, ...]:
        return tuple(self._records)

    def add(self, record: FittedModelRecord) -> FittedModelRecord:
        self._records.append(record)
        self.changed.emit()
        return record

    def create(self, *, artifact: Any = None, **values: Any) -> FittedModelRecord:
        record = FittedModelRecord.create(**values)
        self.add(record)
        if artifact is not None:
            self._artifacts[record.model_id] = artifact
        return record

    def set_artifact(self, model_id: str, artifact: Any) -> None:
        if self.get(model_id) is None:
            raise KeyError(f"Unknown fitted model: {model_id}")
        self._artifacts[str(model_id)] = artifact
        self.changed.emit()

    def get_artifact(self, model_id: str) -> Any:
        return self._artifacts.get(str(model_id))

    def artifacts_dict(self) -> dict[str, Any]:
        return dict(self._artifacts)

    def replace_artifacts(self, artifacts: dict[str, Any] | None) -> None:
        valid_ids = {record.model_id for record in self._records}
        self._artifacts = {
            str(model_id): artifact
            for model_id, artifact in dict(artifacts or {}).items()
            if str(model_id) in valid_ids
        }
        self.changed.emit()

    def get(self, model_id: str) -> FittedModelRecord | None:
        clean_id = str(model_id)
        return next((record for record in self._records if record.model_id == clean_id), None)

    def rename(self, model_id: str, new_name: str) -> FittedModelRecord:
        clean_name = str(new_name).strip()
        if not clean_name:
            raise ValueError("A model name is required.")
        for index, record in enumerate(self._records):
            if record.model_id == str(model_id):
                updated = replace(record, name=clean_name)
                self._records[index] = updated
                self.changed.emit()
                return updated
        raise KeyError(f"Unknown fitted model: {model_id}")

    def remove(self, model_id: str) -> FittedModelRecord:
        for index, record in enumerate(self._records):
            if record.model_id == str(model_id):
                removed = self._records.pop(index)
                self._artifacts.pop(removed.model_id, None)
                self.changed.emit()
                return removed
        raise KeyError(f"Unknown fitted model: {model_id}")

    def clear(self) -> None:
        if self._records:
            self._records.clear()
            self._artifacts.clear()
            self.changed.emit()

    def replace_from_dicts(self, records: list[dict[str, Any]]) -> None:
        self._records = [
            FittedModelRecord.from_dict(item)
            for item in records
            if isinstance(item, dict)
        ]
        valid_ids = {record.model_id for record in self._records}
        self._artifacts = {key: value for key, value in self._artifacts.items() if key in valid_ids}
        self.changed.emit()

    def to_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records]
