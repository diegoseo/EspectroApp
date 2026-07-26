"""Page for reviewing and managing fitted models stored in a project."""

from __future__ import annotations

import json
import re
from typing import Callable

import numpy as np
import pandas as pd

from functions import prepare_pca_matrix
from core.preprocessing_signature import dataset_pipeline_metadata, describe_pipeline
from core.preprocessing_executor import apply_preprocessing_pipeline

from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d

from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)


class PCAProjectionDialog(QDialog):
    """Show training and projected samples in the same saved PCA space."""

    def __init__(
        self,
        *,
        training_scores: np.ndarray,
        projected_scores: np.ndarray,
        training_labels: list[str],
        projected_labels: list[str],
        projected_names: list[str],
        model_name: str,
        translator: Callable[..., str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.tr = translator
        self.model_name = str(model_name)
        self.setWindowTitle(self.tr("PCA projection"))
        self.resize(1100, 800)

        self.training_scores = np.asarray(training_scores, dtype=float)
        self.projected_scores = np.asarray(projected_scores, dtype=float)
        self.training_labels = [str(value) for value in training_labels]
        self.projected_labels = [str(value) for value in projected_labels]
        self.projected_names = [str(value) for value in projected_names]

        self.projected_annotations = []
        self.reference_artist = None
        self.projected_artist = None
        self.hover_annotation = None
        self.axes = None
        self.legend = None

        layout = QVBoxLayout(self)

        title = QLabel(
            self.tr(
                "Projection using the saved PCA model: {name}",
                name=self.model_name,
            )
        )
        title.setObjectName("welcomeTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        note = QLabel(
            self.tr(
                "The PCA model was not refitted. Training and selected samples are displayed in the same component space."
            )
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        controls = QHBoxLayout()

        controls.addWidget(QLabel(self.tr("Projection view")))
        self.view_combo = QComboBox()
        self.view_combo.addItem("PC1 × PC2", ("2d", 0, 1))
        if self.training_scores.shape[1] >= 3 and self.projected_scores.shape[1] >= 3:
            self.view_combo.addItem("PC1 × PC3", ("2d", 0, 2))
            self.view_combo.addItem("PC2 × PC3", ("2d", 1, 2))
            self.view_combo.addItem("PC1 × PC2 × PC3", ("3d", 0, 1, 2))
        self.view_combo.currentIndexChanged.connect(self._build_plot)
        controls.addWidget(self.view_combo)

        self.show_names_checkbox = QCheckBox(
            self.tr("Show projected sample names")
        )
        self.show_names_checkbox.setChecked(True)
        self.show_names_checkbox.toggled.connect(
            self._set_projected_names_visible
        )
        controls.addWidget(self.show_names_checkbox)

        legend_hint = QLabel(
            self.tr("Drag the legend with the mouse to move it.")
        )
        legend_hint.setWordWrap(True)
        controls.addWidget(legend_hint, 1)

        self.export_button = QPushButton(self.tr("Export plot"))
        self.export_button.clicked.connect(self.export_plot)
        controls.addWidget(self.export_button)
        layout.addLayout(controls)

        self.figure = Figure(figsize=(9.4, 6.0), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        self.summary = QLabel()
        layout.addWidget(self.summary)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._build_plot()

    def _current_view(self):
        data = self.view_combo.currentData()
        return data if data is not None else ("2d", 0, 1)

    def _component_name(self, index: int) -> str:
        return f"PC{index + 1}"

    def _build_plot(self) -> None:
        """Build the selected 2D or 3D PCA projection."""
        self.figure.clear()
        mode = self._current_view()
        is_3d = mode[0] == "3d"

        if is_3d:
            self.axes = self.figure.add_subplot(111, projection="3d")
            _, x_index, y_index, z_index = mode
        else:
            self.axes = self.figure.add_subplot(111)
            _, x_index, y_index = mode
            z_index = None

        train = self.training_scores
        projected = self.projected_scores
        required_index = max(value for value in mode[1:])

        if (
            train.ndim != 2
            or projected.ndim != 2
            or train.shape[1] <= required_index
            or projected.shape[1] <= required_index
        ):
            raise ValueError(
                self.tr(
                    "The selected PCA components are not available in this model."
                )
            )

        if is_3d:
            self.reference_artist = self.axes.scatter(
                train[:, x_index],
                train[:, y_index],
                train[:, z_index],
                s=48,
                alpha=0.72,
                label=self.tr("Reference samples"),
            )
            self.projected_artist = self.axes.scatter(
                projected[:, x_index],
                projected[:, y_index],
                projected[:, z_index],
                s=105,
                marker="X",
                edgecolors="black",
                linewidths=0.75,
                label=self.tr("Projected samples"),
            )
            self.axes.set_zlabel(self._component_name(z_index))
        else:
            self.reference_artist = self.axes.scatter(
                train[:, x_index],
                train[:, y_index],
                s=48,
                alpha=0.72,
                label=self.tr("Reference samples"),
            )
            self.projected_artist = self.axes.scatter(
                projected[:, x_index],
                projected[:, y_index],
                s=105,
                marker="X",
                edgecolors="black",
                linewidths=0.75,
                label=self.tr("Projected samples"),
            )

        self.projected_annotations = []
        if not is_3d and projected.shape[0] <= 50:
            for index, sample_name in enumerate(self.projected_names):
                annotation = self.axes.annotate(
                    sample_name,
                    (projected[index, x_index], projected[index, y_index]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    annotation_clip=True,
                    visible=self.show_names_checkbox.isChecked(),
                )
                self.projected_annotations.append(annotation)

        self.axes.set_xlabel(self._component_name(x_index))
        self.axes.set_ylabel(self._component_name(y_index))
        self.axes.set_title(self.tr("Training samples and projected samples"))
        self.axes.grid(True, alpha=0.25)

        self.legend = self.axes.legend(loc="best", fontsize=9)
        if self.legend is not None:
            self.legend.set_draggable(True, use_blit=not is_3d)

        self.hover_annotation = self.axes.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "alpha": 0.94},
            arrowprops={"arrowstyle": "->"},
            fontsize=9,
            visible=False,
        ) if not is_3d else None

        component_text = " × ".join(
            self._component_name(index) for index in mode[1:]
        )
        self.summary.setText(
            self.tr(
                "Projected samples: {count}. Components displayed: {components}.",
                count=projected.shape[0],
                components=component_text,
            )
        )
        self.canvas.draw_idle()

    def _tooltip_text(
        self,
        *,
        name: str,
        group: str,
        score_row: np.ndarray,
        component_indices: tuple[int, ...],
    ) -> str:
        values = [
            f"{self._component_name(index)}: {score_row[index]:.6f}"
            for index in component_indices
        ]
        return "\n".join(
            [
                str(name),
                self.tr("Type: {group}", group=group),
                *values,
            ]
        )

    def _on_hover(self, event) -> None:
        """Show the closest sample name and PCA scores under the pointer."""
        if event.inaxes is not self.axes:
            if self.hover_annotation is not None and self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        mode = self._current_view()
        if mode[0] == "3d":
            self._on_hover_3d(event, mode[1:])
        else:
            self._on_hover_2d(event, mode[1:])

    def _on_hover_2d(self, event, component_indices) -> None:
        x_index, y_index = component_indices
        candidates = []

        for artist, scores, names, group in (
            (
                self.projected_artist,
                self.projected_scores,
                self.projected_names,
                self.tr("Projected sample"),
            ),
            (
                self.reference_artist,
                self.training_scores,
                self.training_labels,
                self.tr("Reference sample"),
            ),
        ):
            contains, details = artist.contains(event)
            if contains and details.get("ind"):
                point_index = int(details["ind"][0])
                candidates.append((scores, names, group, point_index))

        if not candidates:
            if self.hover_annotation is not None and self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        scores, names, group, point_index = candidates[0]
        name = names[point_index] if point_index < len(names) else str(point_index + 1)
        self.hover_annotation.xy = (
            scores[point_index, x_index],
            scores[point_index, y_index],
        )
        self.hover_annotation.set_text(
            self._tooltip_text(
                name=name,
                group=group,
                score_row=scores[point_index],
                component_indices=(x_index, y_index),
            )
        )
        self.hover_annotation.set_visible(True)
        self.canvas.draw_idle()

    def _on_hover_3d(self, event, component_indices) -> None:
        """Display a Qt tooltip for the nearest projected 3D screen position."""
        if event.x is None or event.y is None:
            return

        x_index, y_index, z_index = component_indices
        best = None

        for scores, names, group in (
            (
                self.projected_scores,
                self.projected_names,
                self.tr("Projected sample"),
            ),
            (
                self.training_scores,
                self.training_labels,
                self.tr("Reference sample"),
            ),
        ):
            xs, ys, _ = proj3d.proj_transform(
                scores[:, x_index],
                scores[:, y_index],
                scores[:, z_index],
                self.axes.get_proj(),
            )
            display_points = self.axes.transData.transform(
                np.column_stack([xs, ys])
            )
            distances = np.hypot(
                display_points[:, 0] - event.x,
                display_points[:, 1] - event.y,
            )
            index = int(np.argmin(distances))
            distance = float(distances[index])
            if best is None or distance < best[0]:
                best = (distance, scores, names, group, index)

        if best is None or best[0] > 12:
            self.setToolTip("")
            return

        _, scores, names, group, point_index = best
        name = names[point_index] if point_index < len(names) else str(point_index + 1)
        self.setToolTip(
            self._tooltip_text(
                name=name,
                group=group,
                score_row=scores[point_index],
                component_indices=(x_index, y_index, z_index),
            )
        )

    def _set_projected_names_visible(self, visible: bool) -> None:
        # Text labels are used only in 2D. Hover information remains available
        # in every view, including 3D.
        for annotation in self.projected_annotations:
            annotation.set_visible(bool(visible))
        self.canvas.draw_idle()

    def export_plot(self) -> None:
        """Export the current projection as PNG, PDF, or SVG."""
        default_name = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            f"pca_projection_{self.model_name}",
        ).strip("_") or "pca_projection"

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            self.tr("Export PCA projection"),
            f"{default_name}.png",
            self.tr(
                "PNG image (*.png);;PDF document (*.pdf);;"
                "SVG vector image (*.svg)"
            ),
        )
        if not path:
            return

        try:
            if selected_filter.startswith("PDF"):
                extension = ".pdf"
            elif selected_filter.startswith("SVG"):
                extension = ".svg"
            else:
                extension = ".png"

            if not path.lower().endswith(extension):
                path += extension

            self.figure.savefig(
                path,
                dpi=600 if extension == ".png" else None,
                bbox_inches="tight",
            )
            QMessageBox.information(
                self,
                self.tr("Success"),
                self.tr("Plot saved to:\n{path}", path=path),
            )
        except Exception as error:
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr(
                    "The plot could not be saved:\n{error}",
                    error=error,
                ),
            )


class FittedModelsPage(QFrame):
    """Display fitted-model records and allow safe project-level management."""

    def __init__(
        self,
        *,
        model_manager,
        method_registry,
        translator: Callable[..., str],
        on_project_modified: Callable[[], None] | None = None,
        on_back: Callable[[], None] | None = None,
        datasets_provider: Callable[[], tuple[list, list[str]]] | None = None,
        on_dataset_created: Callable[[pd.DataFrame, str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.model_manager = model_manager
        self.method_registry = method_registry
        self.tr = translator
        self.on_project_modified = on_project_modified
        self.on_back = on_back
        self.datasets_provider = datasets_provider
        self.on_dataset_created = on_dataset_created

        self.setObjectName("workspacePage")
        self._build_ui()
        self.model_manager.changed.connect(self.refresh)
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(14)

        top_row = QHBoxLayout()
        self.back_button = QPushButton(self.tr("Back"))
        self.back_button.clicked.connect(self._go_back)
        top_row.addWidget(self.back_button)
        top_row.addStretch()
        layout.addLayout(top_row)

        description = QLabel(self.tr("Review, rename, or remove the fitted models saved in this project."))
        description.setObjectName("welcomeDescription")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.description_label = description

        content = QHBoxLayout()
        content.setSpacing(14)

        table_card = QFrame()
        table_card.setObjectName("historyCard")
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(18, 18, 18, 18)
        table_layout.setSpacing(10)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Name"),
                self.tr("Method"),
                self.tr("Dataset"),
                self.tr("Created"),
                self.tr("Reusable artifact"),
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._update_details)
        table_layout.addWidget(self.table, 1)

        button_row = QHBoxLayout()
        self.rename_button = QPushButton(self.tr("Rename"))
        self.delete_button = QPushButton(self.tr("Delete"))
        self.apply_button = QPushButton(self.tr("Apply model"))
        self.rename_button.clicked.connect(self.rename_selected)
        self.delete_button.clicked.connect(self.delete_selected)
        self.apply_button.clicked.connect(self.apply_selected)
        button_row.addWidget(self.rename_button)
        button_row.addWidget(self.delete_button)
        button_row.addWidget(self.apply_button)
        button_row.addStretch()
        table_layout.addLayout(button_row)

        detail_card = QFrame()
        detail_card.setObjectName("historyCard")
        detail_card.setMinimumWidth(350)
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(18, 18, 18, 18)
        detail_layout.setSpacing(10)

        self.detail_title = QLabel(self.tr("Model details"))
        self.detail_title.setObjectName("welcomeTitle")
        detail_layout.addWidget(self.detail_title)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText(self.tr("Select a model to view its parameters and metrics."))
        detail_layout.addWidget(self.details, 1)

        content.addWidget(table_card, 3)
        content.addWidget(detail_card, 2)
        layout.addLayout(content, 1)

    def _selected_record(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        model_id = item.data(Qt.UserRole)
        return self.model_manager.get(model_id)

    def refresh(self) -> None:
        selected_id = None
        selected = self._selected_record()
        if selected is not None:
            selected_id = selected.model_id

        records = list(self.model_manager.records)
        self.table.setRowCount(len(records))

        for row, record in enumerate(records):
            try:
                method_name = self.method_registry.get(record.method_id).name
            except KeyError:
                method_name = record.method_id

            values = [
                record.name,
                method_name,
                record.dataset,
                record.created_at.strftime("%Y-%m-%d %H:%M"),
                self.tr("Yes") if record.artifact_path else self.tr("No"),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, record.model_id)
                self.table.setItem(row, column, item)

            if record.model_id == selected_id:
                self.table.selectRow(row)

        self.table.resizeColumnsToContents()
        if records and self.table.currentRow() < 0:
            self.table.selectRow(0)
        self._update_details()

    def _update_details(self) -> None:
        record = self._selected_record()
        enabled = record is not None
        self.rename_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)
        self.apply_button.setEnabled(bool(record and self.model_manager.get_artifact(record.model_id) is not None))

        if record is None:
            self.details.clear()
            return

        try:
            method = self.method_registry.get(record.method_id)
            method_name = method.name
            category = method.category
        except KeyError:
            method_name = record.method_id
            category = self.tr("Unknown")

        payload = [
            f"{self.tr('Name')}: {record.name}",
            f"{self.tr('Method')}: {method_name}",
            f"{self.tr('Category')}: {category}",
            f"{self.tr('Dataset')}: {record.dataset}",
            f"{self.tr('Created')}: {record.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            self.tr("Parameters"),
            json.dumps(record.parameters, indent=2, ensure_ascii=False) if record.parameters else self.tr("None"),
            "",
            self.tr("Metrics"),
            json.dumps(record.metrics, indent=2, ensure_ascii=False) if record.metrics else self.tr("None"),
            "",
            f"{self.tr('Training preprocessing')}: {describe_pipeline({
                'name': (self.model_manager.get_artifact(record.model_id) or {}).get('training_preprocessing_name', ''),
                'options': (self.model_manager.get_artifact(record.model_id) or {}).get('training_preprocessing', {}),
            })}",
            f"{self.tr('Reusable artifact')}: {record.artifact_path or self.tr('Not available yet')}",
        ]
        self.details.setPlainText("\n".join(payload))

    def rename_selected(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        new_name, accepted = QInputDialog.getText(
            self,
            self.tr("Rename model"),
            self.tr("New model name"),
            text=record.name,
        )
        if not accepted or not new_name.strip():
            return
        self.model_manager.rename(record.model_id, new_name.strip())
        if self.on_project_modified:
            self.on_project_modified()

    def delete_selected(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        answer = QMessageBox.question(
            self,
            self.tr("Delete model"),
            self.tr("Delete the fitted model '{name}'?", name=record.name),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.model_manager.remove(record.model_id)
        if self.on_project_modified:
            self.on_project_modified()

    def _go_back(self) -> None:
        if self.on_back is not None:
            self.on_back()

    def apply_selected(self) -> None:
        record = self._selected_record()
        if record is None:
            return

        artifact = self.model_manager.get_artifact(record.model_id)
        if artifact is None:
            QMessageBox.information(
                self,
                self.tr("Model not reusable yet"),
                self.tr("This record stores the model metadata, but the fitted model artifact has not been saved yet."),
            )
            return

        if record.method_id != "pca":
            QMessageBox.information(
                self,
                self.tr("Apply model"),
                self.tr("This model type cannot be applied in the current phase."),
            )
            return

        if self.datasets_provider is None or self.on_dataset_created is None:
            QMessageBox.warning(self, self.tr("Apply model"), self.tr("No datasets are available."))
            return

        dataframes, names = self.datasets_provider()
        if not dataframes:
            QMessageBox.warning(self, self.tr("Apply model"), self.tr("No datasets are available."))
            return

        string_names = [str(name) for name in names]
        dataset_options = []
        for index, (dataframe, visible_name) in enumerate(zip(dataframes, string_names)):
            dataset_id = str(getattr(dataframe, "attrs", {}).get("dataset_id", ""))
            short_id = dataset_id[:8] if dataset_id else str(index + 1)
            dataset_options.append(f"{visible_name}  [{short_id}]")

        selected_option, accepted = QInputDialog.getItem(
            self,
            self.tr("Apply PCA model"),
            self.tr("Select the dataset to transform"),
            dataset_options,
            0,
            False,
        )
        if not accepted:
            return

        action, accepted = QInputDialog.getItem(
            self,
            self.tr("PCA application result"),
            self.tr("What would you like to do with the projected samples?"),
            [
                self.tr("View projection and create scores dataset"),
                self.tr("View projection only"),
                self.tr("Create scores dataset only"),
            ],
            0,
            False,
        )
        if not accepted:
            return

        selected_index = dataset_options.index(selected_option)
        dataset_name = string_names[selected_index]
        source = dataframes[selected_index]

        try:
            matrix = prepare_pca_matrix(source)
            expected = int(artifact.get("n_features", matrix.shape[1]))
            if matrix.shape[1] != expected:
                raise ValueError(
                    self.tr(
                        "The selected dataset has {actual} variables, but the model expects {expected}.",
                        actual=matrix.shape[1],
                        expected=expected,
                    )
                )

            expected_axis = np.asarray(artifact.get("feature_axis", []), dtype=float)
            if expected_axis.size:
                selected_axis = pd.to_numeric(source.iloc[1:, 0], errors="coerce").dropna().to_numpy(dtype=float)
                if selected_axis.size != expected_axis.size or not np.allclose(
                    selected_axis,
                    expected_axis,
                    rtol=1e-7,
                    atol=1e-9,
                    equal_nan=False,
                ):
                    raise ValueError(
                        self.tr(
                            "The selected dataset does not use the same spectral axis and variable order as the training dataset."
                        )
                    )

            training_pipeline = {
                "name": artifact.get("training_preprocessing_name", ""),
                "options": artifact.get("training_preprocessing", {}),
                "signature": artifact.get("training_preprocessing_signature", ""),
            }
            selected_pipeline = dataset_pipeline_metadata(source)
            expected_signature = str(training_pipeline.get("signature") or "")
            pipeline_applied_automatically = False

            if expected_signature and selected_pipeline["signature"] != expected_signature:
                training_options = dict(training_pipeline.get("options") or {})

                # A saved pipeline can safely be applied automatically only to
                # raw spectra. Applying it over a different processed dataset
                # would double-process the signal and could invalidate results.
                if training_options and selected_pipeline.get("is_raw"):
                    answer = QMessageBox.question(
                        self,
                        self.tr("Preprocessing required"),
                        self.tr(
                            "This PCA model was trained with: {training}. The selected dataset is raw. Apply the training preprocessing automatically and continue?",
                            training=describe_pipeline(training_pipeline),
                        ),
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )
                    if answer != QMessageBox.Yes:
                        return

                    source = apply_preprocessing_pipeline(
                        source,
                        training_options,
                        pipeline_name=str(training_pipeline.get("name") or ""),
                    )
                    source.attrs["automatic_preprocessing_for_model"] = record.model_id
                    selected_pipeline = dataset_pipeline_metadata(source)
                    matrix = prepare_pca_matrix(source)
                    pipeline_applied_automatically = True
                elif not training_options and not selected_pipeline.get("is_raw"):
                    raise ValueError(
                        self.tr(
                            "This PCA model was trained with raw data, but the selected dataset is preprocessed. Select the original raw dataset."
                        )
                    )
                else:
                    raise ValueError(
                        self.tr(
                            "The selected dataset already has a different preprocessing pipeline. To avoid applying preprocessing twice, select its original raw dataset or a dataset prepared with: {training}.",
                            training=describe_pipeline(training_pipeline),
                        )
                    )

                if selected_pipeline["signature"] != expected_signature:
                    raise ValueError(
                        self.tr(
                            "The automatic preprocessing did not reproduce the training pipeline signature."
                        )
                    )

            model = artifact["model"]
            scores = model.transform(matrix)
            if scores.shape[1] < 2:
                raise ValueError(self.tr("At least two PCA components are required to display a projection."))

            # En el formato interno de EspectroApp, los nombres visibles de
            # las muestras están en la primera fila. Las columnas del DataFrame
            # pueden ser índices técnicos (1, 2, 3, ...), por lo que no deben
            # utilizarse como etiquetas del gráfico.
            labels = source.iloc[0, 1:].astype(str).tolist()
            sample_names = labels.copy()

            create_dataset = action in {
                self.tr("View projection and create scores dataset"),
                self.tr("Create scores dataset only"),
            }
            show_projection = action in {
                self.tr("View projection and create scores dataset"),
                self.tr("View projection only"),
            }

            if create_dataset:
                rows = [["Type", *labels]]
                for component_index in range(scores.shape[1]):
                    rows.append(
                        [
                            f"PC{component_index + 1}",
                            *scores[:, component_index].astype(float).tolist(),
                        ]
                    )

                result = pd.DataFrame(rows, columns=["X Axis", *sample_names])
                result.attrs = dict(getattr(source, "attrs", {}))
                result.attrs["generated_by_model"] = record.model_id
                result.attrs["source_dataset"] = str(dataset_name)
                result.attrs["pca_model_name"] = record.name
                result.attrs["projection_only"] = True
                result.attrs["automatic_preprocessing_applied"] = pipeline_applied_automatically
                result.attrs["preprocessing_pipeline"] = selected_pipeline.get("options", {})
                result.attrs["preprocessing_signature"] = selected_pipeline.get("signature", "")
                result.attrs["preprocessing_pipeline_name"] = selected_pipeline.get("name", "")
                output_name = f"PCA scores — {dataset_name} — {record.name}"
                self.on_dataset_created(result, output_name)

            if show_projection:
                training_scores = artifact.get("training_scores")
                training_labels = list(artifact.get("training_labels") or [])

                # Prefer the immutable snapshot stored with the PCA model. For
                # legacy models, locate the training dataset by stable ID and
                # only then fall back to its visible name.
                if training_scores is None:
                    expected_id = str(artifact.get("training_dataset_id") or "")
                    training_index = None
                    if expected_id:
                        for index, dataframe in enumerate(dataframes):
                            if str(getattr(dataframe, "attrs", {}).get("dataset_id", "")) == expected_id:
                                training_index = index
                                break
                    if training_index is None:
                        matching = [i for i, name in enumerate(string_names) if name == record.dataset]
                        if len(matching) == 1:
                            training_index = matching[0]

                    if training_index is not None:
                        training_source = dataframes[training_index]
                        training_matrix = prepare_pca_matrix(training_source)
                        if training_matrix.shape[1] != expected:
                            raise ValueError(
                                self.tr("The stored training dataset is no longer compatible with this PCA model.")
                            )
                        training_scores = model.transform(training_matrix)
                        training_labels = training_source.iloc[0, 1:].astype(str).tolist()

                if training_scores is None or len(training_labels) != np.asarray(training_scores).shape[0]:
                    QMessageBox.warning(
                        self,
                        self.tr("Training dataset unavailable"),
                        self.tr("This legacy model does not contain a training snapshot. Refit the PCA once to enable the comparative projection."),
                    )
                else:
                    dialog = PCAProjectionDialog(
                        training_scores=np.asarray(training_scores),
                        projected_scores=scores,
                        training_labels=training_labels,
                        projected_labels=labels,
                        projected_names=sample_names,
                        model_name=record.name,
                        translator=self.tr,
                        parent=self,
                    )
                    dialog.exec()

        except Exception as error:
            QMessageBox.critical(
                self,
                self.tr("Apply model error"),
                self.tr("The model could not be applied:\n{error}", error=error),
            )
            return

        if create_dataset:
            message = self.tr("The PCA projection was created and a scores dataset was added to the project.")
        else:
            message = self.tr("The PCA projection was created using the saved model.")
        if pipeline_applied_automatically:
            message += "\n\n" + self.tr(
                "The training preprocessing pipeline was applied automatically before the PCA projection."
            )
        QMessageBox.information(self, self.tr("Model applied"), message)

