"""Generic method definitions and central registry for EspectroApp."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


MethodRunner = Callable[..., Any]


@dataclass(frozen=True)
class MethodDefinition:
    """Metadata describing one method available in EspectroApp."""

    method_id: str
    name: str
    category: str
    description: str = ""
    runner: MethodRunner | None = None
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    requirements: dict[str, Any] = field(default_factory=dict)
    produces_model: bool = False
    produces_dataset: bool = False
    produces_figure: bool = False
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def normalized_id(self) -> str:
        return self.method_id.strip().lower()


class MethodRegistry:
    """Register and query analysis methods using stable identifiers."""

    def __init__(self) -> None:
        self._methods: dict[str, MethodDefinition] = {}
        self._aliases: dict[str, str] = {}

    def register(self, definition: MethodDefinition, *, replace: bool = False) -> None:
        method_id = definition.normalized_id()
        if not method_id:
            raise ValueError("A method identifier is required.")
        if method_id in self._methods and not replace:
            raise ValueError(f"The method '{method_id}' is already registered.")

        self._methods[method_id] = definition
        self._aliases[method_id] = method_id
        self._aliases[definition.name.strip().lower()] = method_id
        for alias in definition.aliases:
            clean_alias = str(alias).strip().lower()
            if clean_alias:
                self._aliases[clean_alias] = method_id

    def get(self, method_id: str) -> MethodDefinition:
        key = str(method_id).strip().lower()
        canonical = self._aliases.get(key, key)
        try:
            return self._methods[canonical]
        except KeyError as error:
            raise KeyError(f"The method '{method_id}' is not registered.") from error

    def find(self, text: str) -> MethodDefinition | None:
        """Resolve a method from an id, name, alias or operation description."""
        normalized = str(text).strip().lower()
        if not normalized:
            return None
        direct = self._aliases.get(normalized)
        if direct:
            return self._methods[direct]

        candidates: list[tuple[int, MethodDefinition]] = []
        for alias, method_id in self._aliases.items():
            if alias and alias in normalized:
                candidates.append((len(alias), self._methods[method_id]))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def list_all(self) -> tuple[MethodDefinition, ...]:
        return tuple(self._methods.values())

    def list_by_category(self, category: str) -> tuple[MethodDefinition, ...]:
        normalized = str(category).strip().lower()
        return tuple(
            definition
            for definition in self._methods.values()
            if definition.category.strip().lower() == normalized
        )

    def model_methods(self) -> tuple[MethodDefinition, ...]:
        return tuple(item for item in self._methods.values() if item.produces_model)

    def __contains__(self, method_id: object) -> bool:
        if not isinstance(method_id, str):
            return False
        try:
            self.get(method_id)
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        return len(self._methods)


def register_many(registry: MethodRegistry, definitions: Iterable[MethodDefinition]) -> None:
    for definition in definitions:
        registry.register(definition)
