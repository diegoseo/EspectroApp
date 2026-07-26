from datetime import datetime

import pytest


def test_default_registry_contains_expected_methods():
    from methods.defaults import create_default_method_registry

    registry = create_default_method_registry()

    assert len(registry) == 9
    assert "pca" in registry
    assert "PCA analysis" in registry
    assert registry.get("principal components").method_id == "pca"
    assert registry.find("Run low-level fusion now").method_id == "data_fusion"
    assert registry.find("") is None
    assert registry.find("unknown operation") is None

    model_ids = {item.method_id for item in registry.model_methods()}
    assert {"pca", "knn", "svm", "random_forest", "pls"} <= model_ids

    exploration = registry.list_by_category("EXPLORATION")
    assert {item.method_id for item in exploration} == {"pca", "tsne"}


def test_registry_registration_replacement_and_errors():
    from methods.registry import MethodDefinition, MethodRegistry, register_many

    registry = MethodRegistry()
    first = MethodDefinition(
        method_id=" Demo ",
        name="Demo method",
        category="test",
        aliases=("demo alias", ""),
        produces_figure=True,
    )
    registry.register(first)

    assert len(registry) == 1
    assert registry.get("demo") is first
    assert registry.get("DEMO METHOD") is first
    assert registry.get("demo alias") is first
    assert "demo alias" in registry
    assert 123 not in registry

    with pytest.raises(ValueError):
        registry.register(first)

    replacement = MethodDefinition(
        method_id="demo",
        name="Replacement",
        category="other",
        produces_model=True,
    )
    registry.register(replacement, replace=True)
    assert registry.get("demo") is replacement

    with pytest.raises(ValueError):
        registry.register(MethodDefinition("", "Empty", "test"))

    with pytest.raises(KeyError):
        registry.get("missing")

    register_many(
        registry,
        [
            MethodDefinition("second", "Second", "test"),
            MethodDefinition("third", "Third", "test"),
        ],
    )
    assert [item.method_id for item in registry.list_all()] == [
        "demo",
        "second",
        "third",
    ]


def test_fitted_model_record_roundtrip_and_defaults():
    from methods.models import FittedModelRecord

    record = FittedModelRecord.create(
        method_id=" PCA ",
        name="  ",
        dataset=" reference.csv ",
        parameters={"components": 3},
        metrics={"accuracy": 99.3},
        artifact_path="/tmp/model.pkl",
    )

    assert record.method_id == "pca"
    assert record.name.strip().lower() == "pca"
    assert record.dataset == "reference.csv"
    assert record.parameters["components"] == 3

    payload = record.to_dict()
    restored = FittedModelRecord.from_dict(payload)

    assert restored.model_id == record.model_id
    assert restored.created_at == record.created_at
    assert restored.artifact_path == "/tmp/model.pkl"

    fallback = FittedModelRecord.from_dict(
        {
            "method_id": "svm",
            "created_at": "invalid",
            "parameters": None,
            "metrics": None,
        }
    )
    assert fallback.name == "svm"
    assert fallback.dataset == ""
    assert isinstance(fallback.created_at, datetime)


def test_fitted_model_manager_full_lifecycle():
    from methods.models import FittedModelManager, FittedModelRecord

    manager = FittedModelManager()
    changes = []
    manager.changed.connect(lambda: changes.append("changed"))

    first = manager.create(
        method_id="pca",
        name="Reference PCA",
        dataset="reference.csv",
        artifact={"model": object()},
    )
    second = FittedModelRecord.create(
        method_id="hca",
        name="Cluster model",
        dataset="samples.csv",
    )
    manager.add(second)

    assert manager.records == (first, second)
    assert manager.get(first.model_id) == first
    assert manager.get("missing") is None
    assert manager.get_artifact(first.model_id) is not None
    assert first.model_id in manager.artifacts_dict()

    manager.set_artifact(second.model_id, {"tree": 1})
    assert manager.get_artifact(second.model_id) == {"tree": 1}

    with pytest.raises(KeyError):
        manager.set_artifact("missing", object())

    renamed = manager.rename(first.model_id, "Renamed PCA")
    assert renamed.name == "Renamed PCA"

    with pytest.raises(ValueError):
        manager.rename(first.model_id, " ")
    with pytest.raises(KeyError):
        manager.rename("missing", "Name")

    manager.replace_artifacts(
        {
            first.model_id: {"new": 1},
            second.model_id: {"new": 2},
            "unknown": {"ignored": True},
        }
    )
    assert set(manager.artifacts_dict()) == {first.model_id, second.model_id}

    serialized = manager.to_dicts()
    manager.replace_from_dicts(serialized + ["ignored"])
    assert len(manager.records) == 2

    removed = manager.remove(second.model_id)
    assert removed.model_id == second.model_id
    assert manager.get_artifact(second.model_id) is None

    with pytest.raises(KeyError):
        manager.remove("missing")

    manager.clear()
    assert manager.records == ()
    assert manager.artifacts_dict() == {}
    manager.clear()

    assert len(changes) >= 8
