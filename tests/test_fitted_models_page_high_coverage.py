import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def tr(text, **values):
    return str(text).format(**values)


class FakePCAModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)
        self.calls = []

    def transform(self, matrix):
        self.calls.append(np.asarray(matrix))
        return self.scores.copy()


def spectral_df(labels=("A", "B"), dataset_id="new-id"):
    df = pd.DataFrame(
        [
            ["Wavenumber", *labels],
            [100.0, 1.0, 2.0],
            [200.0, 1.5, 2.5],
            [300.0, 2.0, 3.0],
        ]
    )
    df.attrs["dataset_id"] = dataset_id
    df.attrs["data_status"] = "ready"
    return df


def test_projection_dialog_builds_2d_and_3d_and_toggles_names(qapp):
    from ui.pages.fitted_models_page import PCAProjectionDialog

    dialog = PCAProjectionDialog(
        training_scores=np.array([[0, 0, 0], [1, 1, 1]], dtype=float),
        projected_scores=np.array([[2, 2, 2], [3, 3, 3]], dtype=float),
        training_labels=["A", "B"],
        projected_labels=["C", "D"],
        projected_names=["sample-C", "sample-D"],
        model_name="Reference PCA",
        translator=tr,
    )
    try:
        qapp.processEvents()
        assert dialog.view_combo.count() == 4
        assert dialog.axes is not None
        assert len(dialog.projected_annotations) == 2
        assert "Projected samples: 2" in dialog.summary.text()
        assert "PC1 × PC2" in dialog.summary.text()

        dialog.show_names_checkbox.setChecked(False)
        assert all(not annotation.get_visible() for annotation in dialog.projected_annotations)

        dialog.view_combo.setCurrentIndex(3)
        qapp.processEvents()
        assert dialog.axes.name == "3d"
        assert dialog._component_name(2) == "PC3"
        projected_tooltip = dialog._tooltip_text(
            name="sample-C",
            group="C",
            score_row=dialog.projected_scores[0],
            component_indices=(0, 1, 2),
        )
        reference_tooltip = dialog._tooltip_text(
            name="A",
            group="Reference sample",
            score_row=dialog.training_scores[0],
            component_indices=(0, 1, 2),
        )

        assert "sample-C" in projected_tooltip
        assert "Type: C" in projected_tooltip
        assert "A" in reference_tooltip
        assert "PC1:" in reference_tooltip
    finally:
        dialog.close()
        dialog.deleteLater()


def test_projection_dialog_export_success_and_error(qapp, monkeypatch, tmp_path):
    from ui.pages.fitted_models_page import PCAProjectionDialog

    dialog = PCAProjectionDialog(
        training_scores=np.array([[0, 0], [1, 1]], dtype=float),
        projected_scores=np.array([[2, 2]], dtype=float),
        training_labels=["A", "B"],
        projected_labels=["C"],
        projected_names=["sample-C"],
        model_name="PCA",
        translator=tr,
    )
    messages = []
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "projection.png"), "PNG"),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: messages.append(("info", args[-1])),
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda *args, **kwargs: messages.append(("critical", args[-1])),
    )

    try:
        dialog.export_plot()
        assert (tmp_path / "projection.png").exists()
        assert messages[-1][0] == "info"

        monkeypatch.setattr(
            dialog.figure,
            "savefig",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("blocked")),
        )
        dialog.export_plot()
        assert messages[-1][0] == "critical"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_fitted_models_page_refresh_rename_delete_and_back(qapp, monkeypatch):
    from methods.defaults import create_default_method_registry
    from methods.models import FittedModelManager
    from ui.pages.fitted_models_page import FittedModelsPage

    manager = FittedModelManager()
    pca = manager.create(
        method_id="pca",
        name="Original",
        dataset="reference.csv",
        parameters={"components": 3},
        metrics={"accuracy": 99},
        artifact_path="memory",
        artifact={
            "model": FakePCAModel([[1, 2, 3], [4, 5, 6]]),
            "training_preprocessing": {},
        },
    )
    manager.create(
        method_id="unknown",
        name="Unknown",
        dataset="other.csv",
    )

    modified = []
    back = []
    page = FittedModelsPage(
        model_manager=manager,
        method_registry=create_default_method_registry(),
        translator=tr,
        on_project_modified=lambda: modified.append(True),
        on_back=lambda: back.append(True),
    )
    try:
        qapp.processEvents()
        assert page.table.rowCount() == 2
        assert page._selected_record() is not None
        assert "Original" in page.details.toPlainText()
        assert page.apply_button.isEnabled()

        monkeypatch.setattr(
            QInputDialog,
            "getText",
            lambda *args, **kwargs: ("Renamed model", True),
        )
        page.rename_selected()
        assert manager.get(pca.model_id).name == "Renamed model"
        assert modified

        page._go_back()
        assert back == [True]

        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.No,
        )
        page.delete_selected()
        assert len(manager.records) == 2

        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.Yes,
        )
        page.delete_selected()
        assert len(manager.records) == 1
    finally:
        page.close()
        page.deleteLater()


def test_apply_pca_model_creates_scores_dataset(qapp, monkeypatch):
    import ui.pages.fitted_models_page as module
    from methods.defaults import create_default_method_registry
    from methods.models import FittedModelManager

    source = spectral_df()
    model = FakePCAModel([[10, 20], [30, 40]])
    manager = FittedModelManager()
    manager.create(
        method_id="pca",
        name="Reference PCA",
        dataset="reference.csv",
        artifact_path="memory",
        artifact={
            "model": model,
            "n_features": 3,
            "feature_axis": np.array([100.0, 200.0, 300.0]),
            "training_scores": np.array([[0, 0], [1, 1]]),
            "training_labels": ["A", "B"],
            "training_preprocessing": {},
            "training_preprocessing_signature": "",
        },
    )

    created = []
    info = []
    monkeypatch.setattr(
        module,
        "prepare_pca_matrix",
        lambda df: np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )

    choices = iter(
        [
            ("new.csv  [new-id]", True),
            ("Create scores dataset only", True),
        ]
    )
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: info.append(args[-1]),
    )
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: pytest.fail(args[-1]))

    page = module.FittedModelsPage(
        model_manager=manager,
        method_registry=create_default_method_registry(),
        translator=tr,
        datasets_provider=lambda: ([source], ["new.csv"]),
        on_dataset_created=lambda df, name: created.append((df, name)),
    )
    try:
        page.apply_selected()
        assert len(created) == 1
        result, name = created[0]
        assert result.iloc[0, 1:].tolist() == ["A", "B"]
        assert result.shape == (3, 3)
        assert result.attrs["pca_model_name"] == "Reference PCA"
        assert "PCA scores" in name
        assert model.calls
        assert info
    finally:
        page.close()
        page.deleteLater()


def test_apply_model_validation_branches(qapp, monkeypatch):
    import ui.pages.fitted_models_page as module
    from methods.defaults import create_default_method_registry
    from methods.models import FittedModelManager

    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: messages.append(("info", args[-1])),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: messages.append(("warning", args[-1])),
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda *args, **kwargs: messages.append(("critical", args[-1])),
    )

    manager = FittedModelManager()
    manager.create(method_id="pca", name="No artifact", dataset="x")
    page = module.FittedModelsPage(
        model_manager=manager,
        method_registry=create_default_method_registry(),
        translator=tr,
    )
    try:
        page.apply_selected()
        assert messages[-1][0] == "info"
    finally:
        page.close()
        page.deleteLater()

    manager = FittedModelManager()
    manager.create(
        method_id="hca",
        name="Wrong type",
        dataset="x",
        artifact_path="memory",
        artifact={"model": object()},
    )
    page = module.FittedModelsPage(
        model_manager=manager,
        method_registry=create_default_method_registry(),
        translator=tr,
    )
    try:
        page.apply_selected()
        assert messages[-1][0] == "info"
    finally:
        page.close()
        page.deleteLater()
