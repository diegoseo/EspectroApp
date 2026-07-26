import os

"""Preprocessing user interface for EspectroApp.

This module contains the preprocessing page and its related dialogs.
The numerical operations remain in algorithms.preprocessing and are
re-exported through functions.py for backward compatibility.
"""

import os

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtCore import Qt, QTimer
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QMessageBox,
    QLabel,
    QLineEdit,
    QCheckBox,
    QGroupBox,
    QComboBox,
    QButtonGroup,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QFrame,
    QInputDialog,
    QSizePolicy,
)

from thread import PreprocessingThread
from core.pipeline_manager import PipelineManager
from core.preprocessing_signature import canonical_pipeline, pipeline_signature
from core.preprocessing_executor import apply_preprocessing_pipeline
from core.translations import translate, get_language, retranslate_widget_tree
from functions import (
    linear_baseline_from_points,
    shirley_baseline_from_points,
    normalize_by_mean,
    normalize_by_area,
    smooth_savitzky_golay,
    smooth_gaussian_filter,
    smooth_moving_average,
    calculate_first_derivative,
    calculate_second_derivative,
)


def tr(text, **values):
    return translate(text, get_language(), **values)


class DataFrameNameDialog(QDialog):
    """
    Presents a dialog for entering a name to assign to a newly generated data matrix.
    The dialog collects a short descriptive label from the user so the transformed DataFrame can be stored and referenced in the main application.
    """

    def __init__(self):
        super().__init__()

        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("Save Data Matrix"))
        self.setMinimumWidth(460)

        self.setStyleSheet("""
            QDialog {
                background-color: #16212B;
                color: #E5E7EB;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 14px;
            }

            QLabel {
                color: #E5E7EB;
                font-size: 15px;
                font-weight: bold;
                background-color: transparent;
            }

            QLineEdit {
                background-color: #24313D;
                color: #F9FAFB;
                border: 1px solid #3A4B5C;
                border-radius: 8px;
                padding: 9px;
                min-height: 28px;
                font-size: 14px;
            }

            QLineEdit:focus {
                border: 1px solid #5FA8D3;
            }

            QPushButton {
                color: white;
                border-radius: 8px;
                padding: 9px 18px;
                font-weight: bold;
                min-width: 110px;
                min-height: 34px;
            }

            QPushButton#acceptButton {
                background-color: #2F7D4F;
                border: 1px solid #4AA86F;
            }

            QPushButton#acceptButton:hover {
                background-color: #3A9B61;
            }

            QPushButton#cancelButton {
                background-color: #7A3238;
                border: 1px solid #A84A55;
            }

            QPushButton#cancelButton:hover {
                background-color: #9B3D46;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(14)

        label = QLabel(tr("Enter a name for the transformed dataframe:"))
        self.input_nombre = QLineEdit()
        self.input_nombre.setPlaceholderText(tr("E.g.: preprocessed_FTIR"))

        layout.addWidget(label)
        layout.addWidget(self.input_nombre)

        botones_layout = QHBoxLayout()
        botones_layout.setSpacing(10)

        self.accept_button = QPushButton(tr("OK"))
        self.accept_button.setObjectName("acceptButton")
        self.accept_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton(tr("Cancel"))
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.clicked.connect(self.reject)

        botones_layout.addWidget(self.accept_button)
        botones_layout.addWidget(self.cancel_button)

        layout.addLayout(botones_layout)

    def get_name(self):
        return self.input_nombre.text().strip()


class PreprocessingWindow(QWidget):
    """Interactive spectral preprocessing page with live single-spectrum preview."""

    def __init__(self, lista_df, file_names, menu_principal, embedded=False):
        super().__init__()
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.embedded = embedded
        self.menu_principal = menu_principal
        self.lista_df = lista_df.copy()
        self.file_names = file_names
        self.df = None
        self.preview_x = None
        self.preview_y = None
        self.baseline_x_points = []
        self._updating_anchor_lines = False
        self.pipeline_manager = PipelineManager()
        self.active_pipeline_name = None

        if not self.embedded:
            self.setWindowTitle(tr("Spectral Preprocessing"))
            self.setMinimumSize(1200, 760)
            self.resize(1400, 850)

        self.setStyleSheet("""
            QWidget {
                color: #17231D;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 14px;
            }

            QWidget#preprocessingContent {
                background-color: #F8F7F3;
            }

            QGroupBox#mainCard {
                background-color: #FFFFFF;
                border: 1px solid #DEDCD6;
                border-radius: 10px;
                margin-top: 14px;
                padding: 12px;
                font-weight: 600;
            }

            QGroupBox#mainCard::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 7px;
                color: #173D31;
                background-color: #F8F7F3;
                font-weight: 700;
            }

            QGroupBox#filterCard {
                background-color: #FFFFFF;
                border: 1px solid #DEDCD6;
                border-radius: 8px;
                margin-top: 14px;
                padding: 9px;
                font-weight: 600;
            }

            QGroupBox#filterCard::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 9px;
                padding: 0 6px;
                color: #20372D;
                background-color: #FFFFFF;
                font-weight: 600;
            }

            QLabel {
                background-color: transparent;
                color: #24372E;
            }

            QLabel#pointInfo {
                background-color: transparent;
                color: #52655B;
                font-size: 12px;
            }

            QCheckBox,
            QRadioButton {
                background-color: transparent;
                color: #24372E;
                spacing: 7px;
                padding: 3px 1px;
                font-size: 13px;
                font-weight: 500;
            }

            QCheckBox:hover,
            QRadioButton:hover {
                background-color: transparent;
                color: #0F8068;
            }

            QCheckBox::indicator,
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }

            QCheckBox::indicator {
                border: 1px solid #AEB8B2;
                border-radius: 3px;
                background-color: #FFFFFF;
            }

            QCheckBox::indicator:checked {
                background-color: #E66D3C;
                border: 1px solid #E66D3C;
            }

            QRadioButton::indicator {
                border: 1px solid #AEB8B2;
                border-radius: 7px;
                background-color: #FFFFFF;
            }

            QRadioButton::indicator:checked {
                background-color: #E66D3C;
                border: 4px solid #E66D3C;
            }

            QComboBox,
            QLineEdit {
                background-color: #FFFFFF;
                color: #17231D;
                border: 1px solid #D7D7D0;
                border-radius: 6px;
                padding: 6px 8px;
                min-height: 25px;
            }

            QComboBox:hover,
            QLineEdit:hover {
                border: 1px solid #0F8068;
            }

            QComboBox:focus,
            QLineEdit:focus {
                border: 1px solid #0F8068;
            }

            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #17231D;
                selection-background-color: #DDF1EA;
                selection-color: #155D4E;
                border: 1px solid #D7D7D0;
            }

            QPushButton {
                border-radius: 7px;
                padding: 7px 14px;
                min-height: 28px;
                font-weight: 600;
            }

            QPushButton#acceptButton {
                background-color: #0F8068;
                color: #FFFFFF;
                border: 1px solid #0F8068;
            }

            QPushButton#acceptButton:hover {
                background-color: #0B6E59;
            }

            QPushButton#cancelButton {
                background-color: #FFFFFF;
                color: #26322C;
                border: 1px solid #D0D0CA;
            }

            QPushButton#cancelButton:hover {
                background-color: #F1F0EB;
                border: 1px solid #AEB8B2;
            }

            QScrollArea#preprocessingSidebar {
                background-color: #FFFFFF;
                border: none;
            }
            QScrollArea#preprocessingSidebar QWidget#qt_scrollarea_viewport,
            QWidget#preprocessingControls {
                background-color: #FFFFFF;
            }
            QLabel#preprocessingTitle {
                color: #101F1A;
                font-size: 25px;
                font-weight: 750;
                padding-top: 4px;
            }
            QLabel#preprocessingSubtitle {
                color: #4A5E56;
                font-size: 13px;
                padding-bottom: 4px;
            }
            QFrame#compactCard, QFrame#previewCard {
                background-color: #FFFFFF;
                border: 1px solid #D8E0DC;
                border-radius: 9px;
            }
            QLabel#compactSectionTitle {
                color: #10231C;
                font-size: 14px;
                font-weight: 700;
                padding-bottom: 2px;
            }
            QFrame#flatSection {
                background-color: transparent;
                border: none;
                border-top: 1px solid #E1E6E3;
            }
            QLabel#flatSectionTitle {
                color: #10231C;
                font-size: 13px;
                font-weight: 700;
                padding-top: 9px;
            }
            QGroupBox#checkableOption {
                background-color: transparent;
                border: none;
                border-radius: 0px;
                margin-top: 8px;
                padding-top: 6px;
                font-weight: 600;
            }
            QGroupBox#checkableOption::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 25px;
                top: 0px;
                padding: 0 5px 2px 0;
                background-color: transparent;
            }
            QGroupBox#checkableOption::indicator {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 2px;
                top: 0px;
                width: 15px;
                height: 15px;
                border: 1px solid #AEB8B2;
                border-radius: 3px;
                background-color: #FFFFFF;
            }
            QGroupBox#checkableOption::indicator:hover {
                border: 1px solid #0F8068;
            }
            QGroupBox#checkableOption::indicator:checked {
                background-color: #E66D3C;
                border: 1px solid #E66D3C;
            }
            QGroupBox#checkableOption::indicator:disabled {
                background-color: #F1F3F2;
                border: 1px solid #D6DDDA;
            }
            QPushButton#secondaryButton {
                background-color: #FFFFFF;
                color: #26322C;
                border: 1px solid #CAD4CF;
                border-radius: 7px;
                padding: 7px 12px;
                font-weight: 600;
            }
            QPushButton#secondaryButton:hover {
                background-color: #F0F5F3;
                border-color: #86B9AA;
            }
            QPushButton#dangerOutlineButton {
                background-color: #FFFFFF;
                color: #B83F45;
                border: 1px solid #E27A80;
                border-radius: 7px;
                padding: 7px 12px;
                font-weight: 600;
            }
            QPushButton#dangerOutlineButton:hover {
                background-color: #FFF4F4;
                border-color: #C84E55;
            }
            QLabel#previewStatus {
                color: #52655B;
                font-size: 12px;
            }
        """)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # A splitter keeps the preview responsive instead of letting the
        # controls overlap it when the page becomes narrower.
        self.main_splitter = QSplitter(Qt.Horizontal, self)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(6)
        root.addWidget(self.main_splitter)

        # Compact left configuration panel.
        controls_scroll = QScrollArea()
        controls_scroll.setObjectName("preprocessingSidebar")
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(310)
        controls_scroll.setMaximumWidth(440)
        controls_scroll.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding
        )

        controls_widget = QWidget()
        controls_widget.setObjectName("preprocessingControls")
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(18, 10, 18, 18)
        controls_layout.setSpacing(12)

        dataset_group = QFrame()
        dataset_group.setObjectName("compactCard")
        dataset_layout = QGridLayout(dataset_group)
        dataset_layout.setContentsMargins(14, 12, 14, 14)
        dataset_layout.setHorizontalSpacing(10)
        dataset_layout.setVerticalSpacing(7)

        dataset_heading = QLabel(tr("Input dataset"))
        dataset_heading.setObjectName("compactSectionTitle")
        self.selector_df = QComboBox()
        self.selector_df.addItems([os.path.basename(name) for name in self.file_names])
        self.preview_mode = QComboBox()
        self.preview_mode.addItems([
            tr("Single spectrum"),
            tr("All spectra"),
            tr("Average of all samples"),
            tr("Average by class"),
        ])
        self.selector_spectrum = QComboBox()
        self.selector_class = QComboBox()
        self.selector_class.setVisible(False)

        # Stack the selectors vertically. This prevents long translated labels
        # and combo boxes from colliding when the sidebar is narrow.
        dataset_layout.setColumnStretch(0, 1)
        dataset_layout.addWidget(dataset_heading, 0, 0)
        dataset_layout.addWidget(QLabel(tr("Dataset")), 1, 0)
        dataset_layout.addWidget(self.selector_df, 2, 0)
        dataset_layout.addWidget(QLabel(tr("Preview mode")), 3, 0)
        dataset_layout.addWidget(self.preview_mode, 4, 0)
        self.preview_selector_label = QLabel(tr("Preview spectrum"))
        dataset_layout.addWidget(self.preview_selector_label, 5, 0)
        dataset_layout.addWidget(self.selector_spectrum, 6, 0)
        self.class_selector_label = QLabel(tr("Class"))
        self.class_selector_label.setVisible(False)
        dataset_layout.addWidget(self.class_selector_label, 7, 0)
        dataset_layout.addWidget(self.selector_class, 8, 0)

        for selector in (
            self.selector_df,
            self.preview_mode,
            self.selector_spectrum,
            self.selector_class,
        ):
            selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            selector.setMinimumWidth(0)
        controls_layout.addWidget(dataset_group)

        pipeline_group = QFrame()
        pipeline_group.setObjectName("compactCard")
        pipeline_layout = QVBoxLayout(pipeline_group)
        pipeline_layout.setContentsMargins(14, 12, 14, 14)
        pipeline_layout.setSpacing(8)

        pipeline_heading = QLabel(tr("Reusable preprocessing pipeline"))
        pipeline_heading.setObjectName("compactSectionTitle")
        self.pipeline_selector = QComboBox()
        self.pipeline_selector.setPlaceholderText(tr("Select a saved pipeline"))

        self.btn_save_pipeline = QPushButton(tr("Save pipeline"))
        self.btn_load_pipeline = QPushButton(tr("Load pipeline"))
        self.btn_delete_pipeline = QPushButton(tr("Delete pipeline"))
        self.btn_save_pipeline.setObjectName("acceptButton")
        self.btn_load_pipeline.setObjectName("secondaryButton")
        self.btn_delete_pipeline.setObjectName("dangerOutlineButton")

        self.btn_save_pipeline.clicked.connect(self.save_current_pipeline)
        self.btn_load_pipeline.clicked.connect(self.load_selected_pipeline)
        self.btn_delete_pipeline.clicked.connect(self.delete_selected_pipeline)

        pipeline_buttons = QGridLayout()
        pipeline_buttons.setHorizontalSpacing(7)
        pipeline_buttons.setVerticalSpacing(7)
        pipeline_buttons.addWidget(self.btn_save_pipeline, 0, 0, 1, 2)
        pipeline_buttons.addWidget(self.btn_load_pipeline, 1, 0)
        pipeline_buttons.addWidget(self.btn_delete_pipeline, 1, 1)

        pipeline_layout.addWidget(pipeline_heading)
        pipeline_layout.addWidget(self.pipeline_selector)
        pipeline_layout.addLayout(pipeline_buttons)
        controls_layout.addWidget(pipeline_group)

        # Normalization section.
        normalization_group = QFrame()
        normalization_group.setObjectName("flatSection")
        normalization_layout = QVBoxLayout(normalization_group)
        normalization_layout.setContentsMargins(0, 2, 0, 4)
        normalization_layout.setSpacing(7)

        normalization_heading = QLabel(tr("Normalization"))
        normalization_heading.setObjectName("flatSectionTitle")
        normalization_layout.addWidget(normalization_heading)

        self.grupo_normalizar = QGroupBox(tr("Mean normalization"))
        self.grupo_normalizar.setObjectName("checkableOption")
        self.grupo_normalizar.setCheckable(True)
        self.grupo_normalizar.setChecked(False)
        mean_layout = QVBoxLayout(self.grupo_normalizar)
        mean_layout.setContentsMargins(10, 20, 10, 10)
        self.combo_normalizar = QComboBox()
        self.combo_normalizar.addItems(
            [
                "Standardize u=0, v2=1",
                "Center to u=0",
                "Scale to v2=1",
                "Normalize to interval [-1,1]",
                "Normalize to interval [0,1]",
            ]
        )
        mean_layout.addWidget(self.combo_normalizar)
        self.normalizar_a = QCheckBox(tr("Area normalization"))
        normalization_layout.addWidget(self.grupo_normalizar)
        normalization_layout.addWidget(self.normalizar_a)
        controls_layout.addWidget(normalization_group)

        # Smoothing section.
        smoothing_group = QFrame()
        smoothing_group.setObjectName("flatSection")
        smoothing_layout = QVBoxLayout(smoothing_group)
        smoothing_layout.setContentsMargins(0, 2, 0, 4)
        smoothing_layout.setSpacing(7)

        smoothing_heading = QLabel(tr("Smoothing"))
        smoothing_heading.setObjectName("flatSectionTitle")
        smoothing_layout.addWidget(smoothing_heading)

        self.grupo_sg = QGroupBox("Savitzky–Golay")
        self.grupo_sg.setObjectName("checkableOption")
        self.grupo_sg.setCheckable(True)
        self.grupo_sg.setChecked(False)
        sg_layout = QGridLayout(self.grupo_sg)
        sg_layout.setHorizontalSpacing(10)
        sg_layout.setVerticalSpacing(7)
        sg_layout.setContentsMargins(10, 20, 10, 10)
        self.input_ventana_sg = QLineEdit()
        self.input_ventana_sg.setPlaceholderText("5")
        self.input_orden_sg = QLineEdit()
        self.input_orden_sg.setPlaceholderText("2")
        sg_layout.addWidget(QLabel(tr("Window")), 0, 0)
        sg_layout.addWidget(self.input_ventana_sg, 0, 1)
        sg_layout.addWidget(QLabel(tr("Order")), 1, 0)
        sg_layout.addWidget(self.input_orden_sg, 1, 1)

        self.grupo_fg = QGroupBox(tr("Gaussian"))
        self.grupo_fg.setObjectName("checkableOption")
        self.grupo_fg.setCheckable(True)
        self.grupo_fg.setChecked(False)
        fg_layout = QGridLayout(self.grupo_fg)
        fg_layout.setHorizontalSpacing(10)
        fg_layout.setVerticalSpacing(7)
        fg_layout.setContentsMargins(10, 20, 10, 10)
        self.input_sigma_fg = QLineEdit()
        self.input_sigma_fg.setPlaceholderText("2.0")
        fg_layout.addWidget(QLabel("Sigma"), 0, 0)
        fg_layout.addWidget(self.input_sigma_fg, 0, 1)

        self.grupo_mm = QGroupBox(tr("Moving average"))
        self.grupo_mm.setObjectName("checkableOption")
        self.grupo_mm.setCheckable(True)
        self.grupo_mm.setChecked(False)
        mm_layout = QGridLayout(self.grupo_mm)
        mm_layout.setHorizontalSpacing(10)
        mm_layout.setVerticalSpacing(7)
        mm_layout.setContentsMargins(10, 20, 10, 10)
        self.input_ventana_mm = QLineEdit()
        self.input_ventana_mm.setPlaceholderText("3")
        mm_layout.addWidget(QLabel(tr("Window")), 0, 0)
        mm_layout.addWidget(self.input_ventana_mm, 0, 1)

        smoothing_layout.addWidget(self.grupo_sg)
        smoothing_layout.addWidget(self.grupo_fg)
        smoothing_layout.addWidget(self.grupo_mm)
        controls_layout.addWidget(smoothing_group)

        derivative_group = QFrame()
        derivative_group.setObjectName("flatSection")
        derivative_layout = QVBoxLayout(derivative_group)
        derivative_layout.setContentsMargins(0, 2, 0, 4)
        derivative_layout.setSpacing(6)
        derivative_heading = QLabel(tr("Derivatives"))
        derivative_heading.setObjectName("flatSectionTitle")
        derivative_layout.addWidget(derivative_heading)

        self.derivative_none = QRadioButton(tr("None"))
        self.derivada_pd = QRadioButton(tr("First derivative"))
        self.derivada_sd = QRadioButton(tr("Second derivative"))
        self.derivative_none.setChecked(True)
        self.derivative_buttons = QButtonGroup(self)
        for button in (self.derivative_none, self.derivada_pd, self.derivada_sd):
            self.derivative_buttons.addButton(button)
            derivative_layout.addWidget(button)
        controls_layout.addWidget(derivative_group)

        # Baseline correction remains available, but lower in the compact panel.
        baseline_group = QFrame()
        baseline_group.setObjectName("flatSection")
        baseline_layout = QVBoxLayout(baseline_group)
        baseline_layout.setContentsMargins(0, 2, 0, 4)
        baseline_layout.setSpacing(6)
        baseline_heading = QLabel(tr("Baseline correction"))
        baseline_heading.setObjectName("flatSectionTitle")
        baseline_layout.addWidget(baseline_heading)

        self.baseline_none = QRadioButton(tr("None"))
        self.baseline_linear = QRadioButton(
            tr("Linear — select or drag two points on preview")
        )
        self.baseline_shirley = QRadioButton(
            tr("Shirley — select or drag two interval limits")
        )
        self.baseline_none.setChecked(True)
        self.baseline_buttons = QButtonGroup(self)
        for button in (self.baseline_none, self.baseline_linear, self.baseline_shirley):
            self.baseline_buttons.addButton(button)
            baseline_layout.addWidget(button)

        point_row = QHBoxLayout()
        self.point_1_label = QLabel(tr("Point 1: —"))
        self.point_2_label = QLabel(tr("Point 2: —"))
        self.point_1_label.setObjectName("pointInfo")
        self.point_2_label.setObjectName("pointInfo")
        self.reset_points_button = QPushButton(tr("Reset points"))
        self.reset_points_button.setObjectName("secondaryButton")
        self.reset_points_button.clicked.connect(self.reset_baseline_points)
        point_row.addWidget(self.point_1_label)
        point_row.addWidget(self.point_2_label)
        point_row.addStretch()
        point_row.addWidget(self.reset_points_button)
        baseline_layout.addLayout(point_row)

        self.shirley_parameters = QFrame()
        shirley_layout = QGridLayout(self.shirley_parameters)
        shirley_layout.setContentsMargins(0, 4, 0, 0)
        self.input_shirley_tolerance = QLineEdit()
        self.input_shirley_tolerance.setPlaceholderText("1e-6")
        self.input_shirley_iterations = QLineEdit()
        self.input_shirley_iterations.setPlaceholderText("100")
        shirley_layout.addWidget(QLabel(tr("Tolerance")), 0, 0)
        shirley_layout.addWidget(self.input_shirley_tolerance, 0, 1)
        shirley_layout.addWidget(QLabel(tr("Max. iterations")), 1, 0)
        shirley_layout.addWidget(self.input_shirley_iterations, 1, 1)
        self.shirley_parameters.setVisible(False)
        baseline_layout.addWidget(self.shirley_parameters)

        self.baseline_instruction = QLabel(
            tr(
                "Choose Linear or Shirley, then click twice on the graph. "
                "After both limits appear, drag the vertical lines to refine them."
            )
        )
        self.baseline_instruction.setObjectName("pointInfo")
        self.baseline_instruction.setWordWrap(True)
        baseline_layout.addWidget(self.baseline_instruction)
        controls_layout.addWidget(baseline_group)
        controls_layout.addStretch()
        controls_scroll.setWidget(controls_widget)

        # Right live preview panel.
        preview_group = QFrame()
        preview_group.setObjectName("previewCard")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(14, 12, 14, 14)
        preview_layout.setSpacing(8)

        preview_title = QLabel(tr("Live spectral preview"))
        preview_title.setObjectName("compactSectionTitle")
        preview_layout.addWidget(preview_title)

        self.preview_plot = pg.PlotWidget()
        self.preview_plot.setMinimumSize(360, 320)
        self.preview_plot.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.preview_plot.setBackground("w")
        self.preview_plot.showGrid(x=True, y=True, alpha=0.2)
        self.preview_plot.addLegend()
        self.original_curve = self.preview_plot.plot(
            pen=pg.mkPen(color=(90, 90, 90), width=1.8), name=tr("Original")
        )
        self.processed_curve = self.preview_plot.plot(
            pen=pg.mkPen(color=(15, 80, 65), width=2.8), name=tr("Processed")
        )
        self.baseline_curve = self.preview_plot.plot(
            pen=pg.mkPen(color=(190, 80, 40), width=2.2, style=Qt.DashLine),
            name=tr("Baseline"),
        )
        self.preview_extra_items = []

        self.anchor_scatter = pg.ScatterPlotItem()
        self.anchor_scatter.setBrush(pg.mkBrush(180, 50, 30))
        self.anchor_scatter.setPen(pg.mkPen(90, 20, 10, width=1.5))
        self.anchor_scatter.setSize(10)
        self.preview_plot.addItem(self.anchor_scatter)

        self.anchor_lines = []
        for anchor_index in range(2):
            line = pg.InfiniteLine(
                angle=90,
                movable=True,
                pen=pg.mkPen((180, 50, 30), width=1.8),
                hoverPen=pg.mkPen((15, 128, 104), width=2.4),
            )
            line.setVisible(False)
            line.sigPositionChangeFinished.connect(
                lambda moved_line, index=anchor_index: self._on_anchor_line_moved(
                    index, moved_line
                )
            )
            self.preview_plot.addItem(line)
            self.anchor_lines.append(line)

        self.preview_plot.scene().sigMouseClicked.connect(self.on_preview_clicked)
        self.preview_status = QLabel(tr("Select a spectrum to preview."))
        self.preview_status.setObjectName("previewStatus")
        self.preview_status.setWordWrap(True)
        preview_layout.addWidget(self.preview_plot, 1)
        preview_layout.addWidget(self.preview_status)

        buttons = QHBoxLayout()
        back = QPushButton(tr("Back") if embedded else tr("Cancel"))
        back.setObjectName("secondaryButton")
        accept = QPushButton(tr("Accept"))
        accept.setObjectName("acceptButton")
        accept.clicked.connect(self.apply_transformations_and_close)
        back.clicked.connect(
            self.menu_principal.show_welcome_page if embedded else self.close
        )
        buttons.addStretch()
        buttons.addWidget(back)
        buttons.addWidget(accept)
        preview_layout.addLayout(buttons)

        preview_group.setMinimumWidth(380)
        preview_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.main_splitter.addWidget(controls_scroll)
        self.main_splitter.addWidget(preview_group)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([360, 760])

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_preview)

        self.selector_df.currentIndexChanged.connect(self.select_dataframe)
        self.selector_spectrum.currentIndexChanged.connect(
            self.on_preview_spectrum_changed
        )
        self.preview_mode.currentIndexChanged.connect(self.on_preview_mode_changed)
        self.selector_class.currentIndexChanged.connect(self.schedule_preview_update)
        self._connect_preview_controls()
        self.baseline_none.toggled.connect(self._on_baseline_mode_changed)
        self.baseline_linear.toggled.connect(self._on_baseline_mode_changed)
        self.baseline_shirley.toggled.connect(self._on_baseline_mode_changed)

        self.refresh_pipeline_selector()

        if self.selector_df.count() > 0:
            self.select_dataframe(0)

    def _connect_preview_controls(self):
        controls = [
            self.baseline_none,
            self.baseline_linear,
            self.baseline_shirley,
            self.input_shirley_tolerance,
            self.input_shirley_iterations,
            self.grupo_normalizar,
            self.combo_normalizar,
            self.normalizar_a,
            self.grupo_sg,
            self.input_ventana_sg,
            self.input_orden_sg,
            self.grupo_fg,
            self.input_sigma_fg,
            self.grupo_mm,
            self.input_ventana_mm,
            self.derivative_none,
            self.derivada_pd,
            self.derivada_sd,
        ]
        for control in controls:
            if isinstance(control, QLineEdit):
                control.textChanged.connect(self.schedule_preview_update)
            elif isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self.schedule_preview_update)
            elif isinstance(control, QGroupBox):
                control.toggled.connect(self.schedule_preview_update)
            else:
                control.toggled.connect(self.schedule_preview_update)

    def schedule_preview_update(self, *args):
        self.preview_timer.start(250)

    def select_dataframe(self, index):
        if index < 0 or index >= len(self.lista_df):
            return
        self.df = self.lista_df[index].copy()
        self.baseline_x_points = []
        self.selector_spectrum.blockSignals(True)
        self.selector_spectrum.clear()
        labels = [str(v) for v in self.df.iloc[0, 1:].tolist()]
        self.selector_spectrum.addItems(
            [f"{i + 1}: {label}" for i, label in enumerate(labels)]
        )
        self.selector_class.blockSignals(True)
        self.selector_class.clear()
        self.selector_class.addItems(list(dict.fromkeys(labels)))
        self.selector_class.blockSignals(False)
        if self.selector_spectrum.count() > 0:
            self.selector_spectrum.setCurrentIndex(0)
        self.selector_spectrum.blockSignals(False)
        self.reset_baseline_points(update=False)
        self.preview_plot.enableAutoRange(
            axis="xy",
            enable=True,
        )

        self.preview_plot.autoRange()
        self.update_preview()

    def on_preview_mode_changed(self, *args):
        mode = self.preview_mode.currentIndex()
        individual = mode == 0
        class_average = mode == 3
        self.preview_selector_label.setVisible(individual)
        self.selector_spectrum.setVisible(individual)
        self.class_selector_label.setVisible(class_average)
        self.selector_class.setVisible(class_average)
        # Baseline anchor selection is meaningful only for a single spectrum.
        if not individual and self._baseline_mode_active():
            self.baseline_none.setChecked(True)
        self.preview_plot.enableAutoRange(axis="xy", enable=True)
        self.schedule_preview_update()

    def _clear_preview_extra_items(self):
        for item in self.preview_extra_items:
            try:
                self.preview_plot.removeItem(item)
            except Exception:
                pass
        self.preview_extra_items = []

    def _plot_extra_curve(self, x, y, *, pen, name=None):
        item = self.preview_plot.plot(x, y, pen=pen, name=name)
        self.preview_extra_items.append(item)
        return item

    def on_preview_spectrum_changed(self):
        self.preview_plot.enableAutoRange(
            axis="xy",
            enable=True,
        )
        self.preview_plot.autoRange()
        self.schedule_preview_update()

    def _current_spectrum(self):
        if self.df is None or self.df.empty:
            raise ValueError("No dataset is selected.")
        sample_index = self.selector_spectrum.currentIndex()
        if sample_index < 0:
            raise ValueError("No preview spectrum is selected.")
        x = pd.to_numeric(self.df.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(self.df.iloc[1:, sample_index + 1], errors="coerce").to_numpy(
            dtype=float
        )
        if np.isnan(x).any() or np.isnan(y).any():
            raise ValueError(
                "The selected spectrum contains non-numeric or missing values."
            )
        return x, y

    def _baseline_mode_active(self):
        return self.baseline_linear.isChecked() or self.baseline_shirley.isChecked()

    def _on_baseline_mode_changed(self, *args):
        self.shirley_parameters.setVisible(self.baseline_shirley.isChecked())
        self._sync_anchor_visuals()
        self.schedule_preview_update()

    def on_preview_clicked(self, event):
        if not self._baseline_mode_active() or event.button() != Qt.LeftButton:
            return
        if not self.preview_plot.plotItem.sceneBoundingRect().contains(
            event.scenePos()
        ):
            return

        point = self.preview_plot.plotItem.vb.mapSceneToView(event.scenePos())
        x, _ = self._current_spectrum()
        nearest_x = float(x[int(np.argmin(np.abs(x - float(point.x()))))])

        if len(self.baseline_x_points) < 2:
            self.baseline_x_points.append(nearest_x)
        else:
            # A third click moves whichever existing limit is closer.
            distances = [abs(nearest_x - value) for value in self.baseline_x_points]
            replace_index = int(np.argmin(distances))
            self.baseline_x_points[replace_index] = nearest_x

        self.baseline_x_points = sorted(set(self.baseline_x_points))[:2]

        self._update_point_labels()
        self._sync_anchor_visuals()
        self.update_preview()

    def _on_anchor_line_moved(self, index, line):
        if self._updating_anchor_lines:
            return
        if index >= len(self.baseline_x_points):
            return

        try:
            x, _ = self._current_spectrum()
            nearest_x = float(x[int(np.argmin(np.abs(x - float(line.value()))))])
            self.baseline_x_points[index] = nearest_x
            self.baseline_x_points.sort()
            self._update_point_labels()
            self._sync_anchor_visuals()
            self.update_preview()
        except Exception as error:
            self.preview_status.setText(f"Baseline point could not be moved: {error}")

    def _sync_anchor_visuals(self):
        active = self._baseline_mode_active()

        self._updating_anchor_lines = True
        try:
            for index, line in enumerate(self.anchor_lines):
                visible = active and index < len(self.baseline_x_points)
                line.setVisible(visible)
                if visible:
                    line.setPos(self.baseline_x_points[index])
        finally:
            self._updating_anchor_lines = False

        if not active or not self.baseline_x_points:
            self.anchor_scatter.setData([], [])
            return

        try:
            x, y = self._current_spectrum()
            anchor_y = []
            for x_value in self.baseline_x_points:
                point_index = int(np.argmin(np.abs(x - x_value)))
                anchor_y.append(float(y[point_index]))
            self.anchor_scatter.setData(
                self.baseline_x_points,
                anchor_y,
            )
        except Exception:
            self.anchor_scatter.setData([], [])

    def reset_baseline_points(self, update=True):
        self.baseline_x_points = []
        self._update_point_labels()
        self._sync_anchor_visuals()
        if update:
            self.update_preview()

    def _update_point_labels(self):
        self.point_1_label.setText(
            f"Point 1: {self.baseline_x_points[0]:.4g}"
            if len(self.baseline_x_points) > 0
            else "Point 1: —"
        )
        self.point_2_label.setText(
            f"Point 2: {self.baseline_x_points[1]:.4g}"
            if len(self.baseline_x_points) > 1
            else "Point 2: —"
        )

    def update_preview(self):
        try:
            if self.df is None or self.df.empty:
                raise ValueError("No dataset is selected.")

            self._clear_preview_extra_items()
            options = self._build_options()
            processed_df = apply_preprocessing_pipeline(
                self.df, options, pipeline_name=self.active_pipeline_name or ""
            )
            x = pd.to_numeric(self.df.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)
            original_matrix = self.df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            processed_matrix = processed_df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(x).all() or not np.isfinite(original_matrix).all() or not np.isfinite(processed_matrix).all():
                raise ValueError("The dataset contains non-numeric, missing, or infinite values.")

            mode = self.preview_mode.currentIndex()
            self.original_curve.setVisible(False)
            self.processed_curve.setVisible(False)
            self.baseline_curve.setVisible(False)
            self.baseline_curve.setData([], [])

            if mode == 0:
                sample_index = self.selector_spectrum.currentIndex()
                if sample_index < 0 or sample_index >= original_matrix.shape[1]:
                    raise ValueError("No preview spectrum is selected.")
                y_original = original_matrix[:, sample_index]
                y_processed = processed_matrix[:, sample_index]
                derivative_selected = self.derivada_pd.isChecked() or self.derivada_sd.isChecked()
                if not derivative_selected:
                    self.original_curve.setVisible(True)
                    self.original_curve.setData(x, y_original)
                self.processed_curve.setVisible(True)
                self.processed_curve.setData(x, y_processed)
                self.preview_status.setText(tr("Preview updated. Accept applies the same pipeline to all spectra."))

            elif mode == 1:
                # Thin transparent lines keep large datasets readable.
                for column in range(original_matrix.shape[1]):
                    self._plot_extra_curve(
                        x, original_matrix[:, column],
                        pen=pg.mkPen((120, 120, 120, 45), width=0.8),
                    )
                    self._plot_extra_curve(
                        x, processed_matrix[:, column],
                        pen=pg.mkPen((15, 105, 85, 75), width=0.9),
                    )
                self._plot_extra_curve(
                    x, np.mean(processed_matrix, axis=1),
                    pen=pg.mkPen((10, 65, 52), width=3.0),
                    name=tr("Processed average"),
                )
                self.preview_status.setText(
                    tr("Showing {count} spectra and the processed average.", count=original_matrix.shape[1])
                )

            elif mode == 2:
                self.original_curve.setVisible(True)
                self.processed_curve.setVisible(True)
                self.original_curve.setData(x, np.mean(original_matrix, axis=1))
                self.processed_curve.setData(x, np.mean(processed_matrix, axis=1))
                self.preview_status.setText(
                    tr("Average calculated across {count} samples.", count=original_matrix.shape[1])
                )

            else:
                labels = np.asarray([str(v) for v in self.df.iloc[0, 1:].tolist()])
                selected_class = self.selector_class.currentText()
                mask = labels == selected_class
                if not np.any(mask):
                    raise ValueError("No samples were found for the selected class.")
                self.original_curve.setVisible(True)
                self.processed_curve.setVisible(True)
                self.original_curve.setData(x, np.mean(original_matrix[:, mask], axis=1))
                self.processed_curve.setData(x, np.mean(processed_matrix[:, mask], axis=1))
                self.preview_status.setText(
                    tr("Average for class {class_name}: {count} samples.", class_name=selected_class, count=int(mask.sum()))
                )

            self.preview_plot.enableAutoRange(axis="xy", enable=True)
            self.preview_plot.autoRange()
            self._sync_anchor_visuals()
        except Exception as error:
            self.preview_status.setText(f"Preview unavailable: {error}")

    def refresh_pipeline_selector(self):
        """Reload the list of saved preprocessing pipelines."""
        current_name = self.pipeline_selector.currentText().strip()

        self.pipeline_selector.blockSignals(True)
        self.pipeline_selector.clear()

        pipeline_names = self.pipeline_manager.list_names()
        self.pipeline_selector.addItems(pipeline_names)

        if current_name in pipeline_names:
            self.pipeline_selector.setCurrentText(current_name)

        self.pipeline_selector.blockSignals(False)

        has_pipelines = bool(pipeline_names)
        self.btn_load_pipeline.setEnabled(has_pipelines)
        self.btn_delete_pipeline.setEnabled(has_pipelines)

    def save_current_pipeline(self):
        """Save the currently selected preprocessing configuration."""
        try:
            options = self._build_options()
        except Exception as error:
            QMessageBox.warning(
                self,
                tr("Invalid preprocessing options"),
                str(error),
            )
            return

        if not options:
            QMessageBox.warning(
                self,
                tr("Empty pipeline"),
                tr("Select at least one preprocessing operation."),
            )
            return

        name, accepted = QInputDialog.getText(
            self,
            tr("Save preprocessing pipeline"),
            tr("Pipeline name:"),
        )

        if not accepted:
            return

        name = name.strip()

        if not name:
            QMessageBox.warning(
                self,
                tr("Invalid name"),
                tr("Enter a valid pipeline name."),
            )
            return

        try:
            saved_name = self.pipeline_manager.save(
                name=name,
                options=options,
            )
        except Exception as error:
            QMessageBox.critical(
                self,
                tr("Pipeline error"),
                tr("The pipeline could not be saved:\n{error}", error=error),
            )
            return

        self.active_pipeline_name = saved_name
        self.refresh_pipeline_selector()
        self.pipeline_selector.setCurrentText(saved_name)

        QMessageBox.information(
            self,
            tr("Pipeline saved"),
            tr(
                "The preprocessing pipeline '{pipeline_name}' was saved successfully.",
                pipeline_name=saved_name,
            ),
        )

    def load_selected_pipeline(self):
        """Load a saved pipeline into the preprocessing controls."""
        pipeline_name = self.pipeline_selector.currentText().strip()

        if not pipeline_name:
            QMessageBox.warning(
                self,
                tr("No pipeline selected"),
                tr("Select a saved pipeline first."),
            )
            return

        try:
            pipeline = self.pipeline_manager.load(pipeline_name)
            self.apply_pipeline_to_controls(pipeline.get("options", {}))
        except Exception as error:
            QMessageBox.critical(
                self,
                tr("Pipeline error"),
                tr("The pipeline could not be loaded:\n{error}", error=error),
            )
            return

        self.active_pipeline_name = pipeline_name
        self.schedule_preview_update()

        QMessageBox.information(
            self,
            tr("Pipeline loaded"),
            tr(
                "The pipeline '{pipeline_name}' was loaded. "
                "Review the preview and press Accept to apply it.",
                pipeline_name=pipeline_name,
            ),
        )

    def delete_selected_pipeline(self):
        """Delete the selected saved pipeline."""
        pipeline_name = self.pipeline_selector.currentText().strip()

        if not pipeline_name:
            return

        confirmation_box = QMessageBox(self)
        confirmation_box.setIcon(QMessageBox.Question)
        confirmation_box.setWindowTitle(tr("Delete pipeline"))
        confirmation_box.setText(
            tr(
                "Delete the pipeline '{pipeline_name}'?",
                pipeline_name=pipeline_name,
            )
        )
        confirmation_box.setStandardButtons(
            QMessageBox.Yes | QMessageBox.No
        )
        confirmation_box.setDefaultButton(QMessageBox.No)

        yes_button = confirmation_box.button(QMessageBox.Yes)
        no_button = confirmation_box.button(QMessageBox.No)

        if yes_button is not None:
            yes_button.setText(tr("Yes"))
        if no_button is not None:
            no_button.setText(tr("No"))

        answer = confirmation_box.exec()

        if answer != QMessageBox.Yes:
            return

        try:
            self.pipeline_manager.delete(pipeline_name)
        except Exception as error:
            QMessageBox.critical(
                self,
                tr("Pipeline error"),
                tr("The pipeline could not be deleted:\n{error}", error=error),
            )
            return

        if self.active_pipeline_name == pipeline_name:
            self.active_pipeline_name = None

        self.refresh_pipeline_selector()

    def reset_preprocessing_controls(self):
        """Reset every preprocessing control before loading a pipeline."""
        self.baseline_none.setChecked(True)
        self.baseline_x_points = []
        self.input_shirley_tolerance.clear()
        self.input_shirley_iterations.clear()

        self.grupo_normalizar.setChecked(False)
        self.combo_normalizar.setCurrentIndex(0)
        self.normalizar_a.setChecked(False)

        self.grupo_sg.setChecked(False)
        self.input_ventana_sg.clear()
        self.input_orden_sg.clear()

        self.grupo_fg.setChecked(False)
        self.input_sigma_fg.clear()

        self.grupo_mm.setChecked(False)
        self.input_ventana_mm.clear()

        self.derivative_none.setChecked(True)
        self.reset_baseline_points(update=False)

    def apply_pipeline_to_controls(self, options):
        """Populate the graphical controls from saved pipeline options."""
        self.reset_preprocessing_controls()

        linear = options.get("correccion_lineal")
        shirley = options.get("correccion_shirley")

        if linear:
            self.baseline_linear.setChecked(True)
            self.baseline_x_points = [
                float(linear["x_start"]),
                float(linear["x_end"]),
            ]

        elif shirley:
            self.baseline_shirley.setChecked(True)
            self.baseline_x_points = [
                float(shirley["x_start"]),
                float(shirley["x_end"]),
            ]
            self.input_shirley_tolerance.setText(str(shirley.get("tolerance", "1e-6")))
            self.input_shirley_iterations.setText(
                str(shirley.get("max_iterations", "100"))
            )

        mean_normalization = options.get("normalizar_media")
        if mean_normalization:
            self.grupo_normalizar.setChecked(True)
            method = mean_normalization.get("metodo")
            if method:
                index = self.combo_normalizar.findText(str(method))
                if index >= 0:
                    self.combo_normalizar.setCurrentIndex(index)

        self.normalizar_a.setChecked(bool(options.get("normalizar_area")))

        savgol = options.get("suavizar_sg")
        if savgol:
            self.grupo_sg.setChecked(True)
            self.input_ventana_sg.setText(str(savgol.get("ventana", "")))
            self.input_orden_sg.setText(str(savgol.get("orden", "")))

        gaussian = options.get("suavizar_fg")
        if gaussian:
            self.grupo_fg.setChecked(True)
            self.input_sigma_fg.setText(str(gaussian.get("sigma", "")))

        moving_average = options.get("suavizar_mm")
        if moving_average:
            self.grupo_mm.setChecked(True)
            self.input_ventana_mm.setText(str(moving_average.get("ventana", "")))

        if options.get("derivada_1"):
            self.derivada_pd.setChecked(True)
        elif options.get("derivada_2"):
            self.derivada_sd.setChecked(True)

        self._sync_anchor_visuals()
        self._on_baseline_mode_changed()
        self.update_preview()

    def _build_options(self):
        options = {}

        if self.baseline_linear.isChecked():
            if len(self.baseline_x_points) != 2:
                raise ValueError(
                    "Select two points on the preview "
                    "for linear baseline correction."
                )

            options["correccion_lineal"] = {
                "x_start": float(self.baseline_x_points[0]),
                "x_end": float(self.baseline_x_points[1]),
            }

        elif self.baseline_shirley.isChecked():
            if len(self.baseline_x_points) != 2:
                raise ValueError(
                    "Select two interval limits on the preview "
                    "for Shirley baseline correction."
                )

            tolerance_text = self.input_shirley_tolerance.text().strip() or "1e-6"
            iterations_text = self.input_shirley_iterations.text().strip() or "100"

            try:
                tolerance = float(tolerance_text)
                max_iterations = int(iterations_text)
            except ValueError as error:
                raise ValueError(
                    "Shirley tolerance must be numeric and "
                    "maximum iterations must be an integer."
                ) from error

            if tolerance <= 0:
                raise ValueError("Shirley tolerance must be greater than zero.")
            if max_iterations < 1:
                raise ValueError("Shirley maximum iterations must be at least 1.")

            options["correccion_shirley"] = {
                "x_start": float(self.baseline_x_points[0]),
                "x_end": float(self.baseline_x_points[1]),
                "tolerance": tolerance,
                "max_iterations": max_iterations,
            }

        if self.grupo_normalizar.isChecked():
            options["normalizar_media"] = {
                "activar": True,
                "metodo": self.combo_normalizar.currentText(),
            }

        if self.normalizar_a.isChecked():
            options["normalizar_area"] = True

        if self.grupo_sg.isChecked():
            window_text = self.input_ventana_sg.text().strip()

            order_text = self.input_orden_sg.text().strip()

            if not window_text or not order_text:
                raise ValueError(
                    "Enter the window size and polynomial order " "for Savitzky–Golay."
                )

            try:
                window = int(window_text)
                order = int(order_text)
            except ValueError as error:
                raise ValueError(
                    "Savitzky–Golay window and polynomial order "
                    "must be integer numbers."
                ) from error

            if window < 3:
                raise ValueError("The Savitzky–Golay window must be at least 3.")

            if window % 2 == 0:
                raise ValueError("The Savitzky–Golay window must be odd.")

            if order < 0:
                raise ValueError(
                    "The Savitzky–Golay polynomial order " "cannot be negative."
                )

            if order >= window:
                raise ValueError(
                    "The Savitzky–Golay polynomial order "
                    "must be lower than the window size."
                )

            options["suavizar_sg"] = {
                "ventana": window,
                "orden": order,
            }

        if self.grupo_fg.isChecked():
            sigma_text = self.input_sigma_fg.text().strip()

            if not sigma_text:
                raise ValueError("Enter a sigma value for the Gaussian filter.")

            try:
                sigma = float(sigma_text)
            except ValueError as error:
                raise ValueError("Gaussian sigma must be a numeric value.") from error

            if sigma <= 0:
                raise ValueError("Gaussian sigma must be greater than zero.")

            options["suavizar_fg"] = {
                "sigma": sigma,
            }

        if self.grupo_mm.isChecked():
            window_text = self.input_ventana_mm.text().strip()

            if not window_text:
                raise ValueError("Enter a window size for the moving average.")

            try:
                window = int(window_text)
            except ValueError as error:
                raise ValueError(
                    "Moving-average window must be an integer number."
                ) from error

            if window < 1:
                raise ValueError("Moving-average window must be greater than zero.")

            options["suavizar_mm"] = {
                "ventana": window,
            }

        if self.derivada_pd.isChecked():
            options["derivada_1"] = True

        elif self.derivada_sd.isChecked():
            options["derivada_2"] = True

        return options

    def _selected_source_name(self):
        index = self.selector_df.currentIndex()

        file_names = getattr(
            self,
            "file_names",
            getattr(self, "nombres_archivos", []),
        )

        if index < 0 or index >= len(file_names):
            return "Unnamed dataset"

        return os.path.basename(str(file_names[index]))

    def _history_summary(self, options):
        """Create a readable operation name and parameter dictionary."""
        operations = []
        parameters = {}

        if options.get("normalizar_media"):
            operations.append("Mean normalization")

        if options.get("normalizar_area"):
            operations.append("Area normalization")

        if options.get("centrar"):
            operations.append("Mean centering")

        if options.get("escalar"):
            operations.append("Variance scaling")

        if options.get("estandarizar"):
            operations.append("Standardization")

        interval = options.get("normalizar_intervalo")
        if interval:
            operations.append(f"Normalization to {interval}")

        baseline = options.get("correccion_linea_base")
        if baseline:
            operations.append("Linear baseline correction")
            if isinstance(baseline, dict):
                if "x_start" in baseline:
                    parameters["Baseline start"] = baseline["x_start"]
                if "x_end" in baseline:
                    parameters["Baseline end"] = baseline["x_end"]

        shirley = options.get("correccion_shirley")
        if shirley:
            operations.append("Shirley baseline correction")
            if isinstance(shirley, dict):
                if "x_start" in shirley:
                    parameters["Shirley start"] = shirley["x_start"]
                if "x_end" in shirley:
                    parameters["Shirley end"] = shirley["x_end"]

        savgol = options.get("suavizar_sg")
        if savgol:
            operations.append("Savitzky–Golay smoothing")
            parameters["SG window"] = savgol.get("ventana")
            parameters["SG order"] = savgol.get("orden")

        gaussian = options.get("suavizar_fg")
        if gaussian:
            operations.append("Gaussian smoothing")
            parameters["Gaussian sigma"] = gaussian.get("sigma")

        moving = options.get("suavizar_mm")
        if moving:
            operations.append("Moving-average smoothing")
            parameters["Moving-average window"] = moving.get("ventana")

        if options.get("derivada_1"):
            operations.append("First derivative")

        if options.get("derivada_2"):
            operations.append("Second derivative")

        if not operations:
            operations.append("Spectral preprocessing")

        return " + ".join(operations), parameters

    def apply_transformations_and_close(self):
        self.apply_transformations()

        if not self.embedded:
            self.close()

    def apply_transformations(self):
        try:
            if self.selector_df.currentIndex() < 0:
                raise ValueError("No data matrix was selected.")
            source_df = self.lista_df[self.selector_df.currentIndex()]
            self.df = source_df.copy()
            # Pandas normally preserves attrs on copy, but keep them explicitly
            # because they are essential for reusable PCA compatibility checks.
            self.df.attrs = dict(getattr(source_df, "attrs", {}) or {})
            options = self._build_options()
            self._history_source_name = self._selected_source_name()
            self._history_options = options.copy()

            if self.active_pipeline_name:
                try:
                    saved_pipeline = self.pipeline_manager.load(
                        self.active_pipeline_name
                    )
                    if saved_pipeline.get("options", {}) != options:
                        self.active_pipeline_name = None
                except Exception:
                    self.active_pipeline_name = None
        except Exception as error:
            QMessageBox.warning(
                self,
                tr("Invalid preprocessing options"),
                str(error),
            )
            return

        self.hilo = PreprocessingThread(self.df, options)
        self.hilo.dataframe_result.connect(self.receive_transformed_dataframe)
        self.hilo.start()

    def receive_transformed_dataframe(self, df_transformado):
        dialog = DataFrameNameDialog()
        if dialog.exec():
            name = dialog.get_name()
            if not name:
                QMessageBox.warning(
                self,
                tr("Empty name"),
                tr("Please enter a valid name."),
            )
                return
            # Preserve a complete and comparable preprocessing history.
            source_index = self.selector_df.currentIndex()
            source_df = self.lista_df[source_index] if 0 <= source_index < len(self.lista_df) else None
            source_attrs = dict(getattr(source_df, "attrs", {}) or {})
            options = canonical_pipeline(getattr(self, "_history_options", {}))
            output_attrs = dict(getattr(df_transformado, "attrs", {}) or {})
            output_attrs["preprocessing_pipeline"] = options
            output_attrs["preprocessing_signature"] = pipeline_signature(options)
            output_attrs["preprocessing_pipeline_name"] = self.active_pipeline_name or ""
            output_attrs["preprocessing_applied"] = bool(options)
            operation_text, _ = self._history_summary(options)
            output_attrs["preprocessing_description"] = operation_text
            output_attrs["source_dataset_id"] = str(source_attrs.get("dataset_id", ""))
            output_attrs["source_dataset_name"] = getattr(
                self, "_history_source_name", "Unnamed dataset"
            )
            df_transformado.attrs = output_attrs

            self.menu_principal.dataframes.append(df_transformado)
            self.menu_principal.nombres_archivos.append(name)
            if hasattr(self.menu_principal, "_ensure_dataset_metadata"):
                self.menu_principal._ensure_dataset_metadata()

            if hasattr(self.menu_principal, "record_analysis_step"):
                operation, parameters = self._history_summary(
                    getattr(self, "_history_options", {})
                )

                if self.active_pipeline_name:
                    parameters["Pipeline"] = self.active_pipeline_name
                    operation = f"Pipeline applied: " f"{self.active_pipeline_name}"

                self.menu_principal.record_analysis_step(
                    dataset=getattr(
                        self,
                        "_history_source_name",
                        "Unnamed dataset",
                    ),
                    operation=operation,
                    output_dataset=name,
                    parameters=parameters,
                )

            QMessageBox.information(
                self,
                "Success",
                f"Transformed data matrix saved as '{name}' and added to the current session.",
            )


class PostTransformationOptionsWindow(QWidget):
    """
    Presents follow-up actions for a newly transformed DataFrame in a simple choice window.
    The user can choose to inspect the matrix in tabular form or open the spectra visualization options for further analysis.

    Parameters
    ----------
    menu_principal : QWidget
        Reference to the main menu or controller that provides dataframe viewing and spectra plotting capabilities.
    df_transformado : pandas.DataFrame
        Transformed spectral DataFrame on which the user will perform the next action.
    """

    def __init__(self, menu_principal, df_transformado):
        super().__init__()
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.menu_principal = menu_principal
        self.df = df_transformado

        self.setWindowTitle("Actions for the transformed DataFrame")

        layout = QVBoxLayout()
        layout.addWidget(
            QLabel("What would you like to do with the transformed DataFrame?")
        )

        btn_ver_df = QPushButton("View DataFrame")
        btn_ver_espectro = QPushButton("Display Spectra")

        btn_ver_df.clicked.connect(self.view_dataframe)
        btn_ver_espectro.clicked.connect(self.open_spectra_window)

        layout.addWidget(btn_ver_df)
        layout.addWidget(btn_ver_espectro)
        self.setLayout(layout)

    def view_dataframe(self):
        self.menu_principal.view_dataframe(self.df)
        self.close()

    def open_spectra_window(self):
        self.menu_principal.open_spectra_window(self.df)
        self.close()