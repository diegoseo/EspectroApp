"""Method registration infrastructure for EspectroApp."""

from methods.defaults import create_default_method_registry
from methods.models import FittedModelManager, FittedModelRecord
from methods.registry import MethodDefinition, MethodRegistry

__all__ = [
    "MethodDefinition",
    "MethodRegistry",
    "FittedModelRecord",
    "FittedModelManager",
    "create_default_method_registry",
]
