import os
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, Signal
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QLabel,
    QLineEdit,
    QCheckBox,
    QGroupBox,
    QComboBox,
    QButtonGroup,
    QRadioButton,
    QTabWidget,
    QGridLayout,
    QFrame,
    QSizePolicy,
    QScrollArea,
)


from core.translations import translate, get_language, retranslate_widget_tree


def tr(text, **values):
    return translate(text, get_language(), **values)


class SpectraResultsPage(QWidget):
    def __init__(self, back_callback, parent=None):
        super().__init__(parent)

        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.plot_widgets = {}

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        # Barra superior
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(10)

        self.btn_back = QPushButton(tr("← Back to options"))
        self.btn_back.setObjectName("backButton")
        self.btn_back.clicked.connect(back_callback)

        title_label = QLabel(tr("Spectral visualization results"))
        title_label.setStyleSheet("""
            font-size: 19px;
            font-weight: 700;
            color: #17231D;
            background-color: transparent;
            """)

        toolbar_layout.addWidget(self.btn_back)
        toolbar_layout.addWidget(title_label)
        toolbar_layout.addStretch()

        main_layout.addLayout(toolbar_layout)

        # Pestañas
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        self.tabs.setTabsClosable(False)

        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #D8D8D2;
                border-radius: 8px;
                background-color: #FFFFFF;
            }

            QTabBar::tab {
                background-color: #EAECE8;
                color: #34443C;
                border: 1px solid #D2D7D3;
                border-bottom: none;
                padding: 9px 16px;
                margin-right: 2px;
                font-weight: 600;
            }

            QTabBar::tab:selected {
                background-color: #FFFFFF;
                color: #0F725C;
            }

            QTabBar::tab:hover {
                background-color: #F6F7F5;
            }
            """)

        main_layout.addWidget(self.tabs, 1)

    def add_plot(self, plot_widget, title):
        """
        Adds or replaces a spectral plot tab.
        """

        # Eliminar una pestaña anterior con el mismo nombre
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == title:
                old_widget = self.tabs.widget(index)
                self.tabs.removeTab(index)
                old_widget.deleteLater()
                break

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)

        plot_widget.setParent(plot_container)
        plot_widget.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        plot_layout.addWidget(plot_widget, 1)

        new_index = self.tabs.addTab(
            plot_container,
            title,
        )

        self.plot_widgets[title] = plot_widget
        self.tabs.setCurrentIndex(new_index)


class SpectraExportOptionsWindow(QWidget):

    seleccion_confirmada = Signal(int, object)
    cancel_requested = Signal()

    def __init__(
        self,
        dataframes,
        file_names,
        parent=None,
    ):
        super().__init__(parent)

        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.dataframes = dataframes
        self.rutas_completas = file_names

        self.setMinimumWidth(600)

        self.setStyleSheet("""
            QWidget {
                background-color: #F8F7F3;
                color: #1F2924;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 12px;
            }

            QLabel#sectionLabel {
                color: #2D3832;
                font-size: 14px;
                font-weight: 600;
                background-color: transparent;
            }

            QLabel#parameterLabel {
                color: #5E6B64;
                font-size: 12px;
                font-weight: 600;
                background-color: transparent;
            }

            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #DEDED8;
                border-radius: 10px;
                margin-top: 16px;
                padding: 12px;
            }

            QGroupBox::title {
                color: #68776E;
                font-size: 14px;
                font-weight: 600;
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                background-color: #F8F7F3;
            }

            QComboBox,
            QLineEdit {
                background-color: #FFFFFF;
                color: #1F2924;
                border: 1px solid #D7D7D0;
                border-radius: 8px;
                padding: 8px 11px;
                min-height: 28px;
                font-size: 13px;
            }

            QComboBox:hover,
            QLineEdit:hover {
                border: 1px solid #138269;
            }

            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #1F2924;
                selection-background-color: #DDF1EA;
                selection-color: #126B58;
            }

            QCheckBox,
            QRadioButton {
                color: #26322C;
                font-size: 14px;
                padding: 8px 10px;
                border-radius: 8px;
                background-color: transparent;
                min-height: 22px;
            }

            QCheckBox:hover,
            QRadioButton:hover {
                background-color: #F1F4F1;
            }

            QCheckBox:checked,
            QRadioButton:checked {
                background-color: #DDF1EA;
                color: #163F34;
                font-weight: 600;
            }

            QCheckBox::indicator,
            QRadioButton::indicator {
                width: 17px;
                height: 17px;
                margin-right: 8px;
            }

            QPushButton {
                border-radius: 7px;
                padding: 8px 18px;
                min-height: 28px;
                font-size: 12px;
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
                background-color: #F0F0EC;
            }
            """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(10)

        dataset_group = QGroupBox(tr("Input dataset"))
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.setContentsMargins(18, 20, 18, 14)
        dataset_layout.setSpacing(7)

        dataset_label = QLabel(tr("Choose a data matrix:"))
        dataset_label.setObjectName("sectionLabel")

        self.combo_archivo = QComboBox()
        nombres_visibles = [os.path.basename(path) for path in file_names]
        self.combo_archivo.addItems(nombres_visibles)
        self.combo_archivo.currentIndexChanged.connect(self.update_dataset_parameters)

        dataset_layout.addWidget(dataset_label)
        dataset_layout.addWidget(self.combo_archivo)
        main_layout.addWidget(dataset_group)

        self.options_tabs = QTabWidget()
        self.options_tabs.setDocumentMode(True)

        # Visualization tab
        visualization_tab = QWidget()
        visualization_tab_layout = QHBoxLayout(visualization_tab)
        visualization_tab_layout.setContentsMargins(10, 10, 10, 10)
        visualization_tab_layout.setSpacing(12)

        plot_types_group = QGroupBox(tr("Plot types"))
        plot_types_layout = QVBoxLayout(plot_types_group)
        plot_types_layout.setContentsMargins(16, 20, 16, 14)
        plot_types_layout.setSpacing(5)

        self.check_full_plot = QCheckBox(tr("Full spectra plot"))
        self.check_limited_plot = QCheckBox(tr("Limited-range spectra plot"))
        self.check_type_plot = QCheckBox(tr("Spectra plot by sample type"))
        self.check_limited_type_plot = QCheckBox(
            tr("Limited-range spectra plot by sample type")
        )
        self.check_stacked_plot = QCheckBox(tr("Stacked spectra plot"))
        self.check_full_plot.setChecked(True)

        for checkbox in (
            self.check_full_plot,
            self.check_limited_plot,
            self.check_type_plot,
            self.check_limited_type_plot,
            self.check_stacked_plot,
        ):
            plot_types_layout.addWidget(checkbox)

        plot_types_layout.addStretch()
        visualization_tab_layout.addWidget(plot_types_group, 1)

        configuration_group = QGroupBox(tr("Plot configuration"))
        configuration_layout = QVBoxLayout(configuration_group)
        configuration_layout.setContentsMargins(10, 18, 10, 10)

        configuration_scroll = QScrollArea()
        configuration_scroll.setWidgetResizable(True)
        configuration_scroll.setFrameShape(QFrame.NoFrame)
        configuration_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        configuration_content = QWidget()
        configuration_content_layout = QVBoxLayout(configuration_content)
        configuration_content_layout.setContentsMargins(4, 4, 8, 4)
        configuration_content_layout.setSpacing(10)

        self.configuration_empty = QLabel(
            tr("Select a plot type that requires additional parameters.")
        )
        self.configuration_empty.setWordWrap(True)
        self.configuration_empty.setObjectName("parameterLabel")
        configuration_content_layout.addWidget(self.configuration_empty)

        self.range_group = QGroupBox(tr("X-axis range"))
        range_layout = QGridLayout(self.range_group)
        range_layout.setContentsMargins(14, 20, 14, 12)
        range_layout.setHorizontalSpacing(10)
        range_layout.setVerticalSpacing(8)

        minimum_label = QLabel(tr("Minimum X:"))
        minimum_label.setObjectName("parameterLabel")
        maximum_label = QLabel(tr("Maximum X:"))
        maximum_label.setObjectName("parameterLabel")

        self.input_range_min = QLineEdit()
        self.input_range_max = QLineEdit()
        self.input_range_min.setPlaceholderText(tr("Minimum value"))
        self.input_range_max.setPlaceholderText(tr("Maximum value"))

        range_layout.addWidget(minimum_label, 0, 0)
        range_layout.addWidget(self.input_range_min, 0, 1)
        range_layout.addWidget(maximum_label, 1, 0)
        range_layout.addWidget(self.input_range_max, 1, 1)
        configuration_content_layout.addWidget(self.range_group)

        self.type_group = QGroupBox(tr("Sample type"))
        type_layout = QVBoxLayout(self.type_group)
        type_layout.setContentsMargins(14, 20, 14, 12)
        type_layout.setSpacing(8)

        type_label = QLabel(tr("Choose the sample type to display:"))
        type_label.setObjectName("parameterLabel")
        self.combo_sample_type = QComboBox()

        type_layout.addWidget(type_label)
        type_layout.addWidget(self.combo_sample_type)
        configuration_content_layout.addWidget(self.type_group)

        self.stacked_group = QGroupBox(tr("Stacked spectra settings"))
        stacked_layout = QGridLayout(self.stacked_group)
        stacked_layout.setContentsMargins(14, 20, 14, 12)
        stacked_layout.setHorizontalSpacing(10)
        stacked_layout.setVerticalSpacing(8)

        self.check_stacked_auto_offset = QCheckBox(tr("Automatic vertical offset"))
        self.check_stacked_auto_offset.setChecked(True)

        self.input_stacked_offset = QLineEdit()
        self.input_stacked_offset.setText("1.15")
        self.input_stacked_offset.setPlaceholderText(
            tr("Automatic multiplier or manual offset")
        )

        self.check_stacked_labels = QCheckBox(tr("Show spectrum labels"))
        self.check_stacked_labels.setChecked(True)

        self.input_stacked_max_spectra = QLineEdit()
        self.input_stacked_max_spectra.setText("10")
        self.input_stacked_max_spectra.setPlaceholderText(
            tr("Maximum number of spectra")
        )

        self.check_stacked_by_type = QCheckBox(tr("Show only selected sample type"))
        self.check_stacked_limited = QCheckBox(tr("Use selected X range"))

        offset_label = QLabel(tr("Offset value:"))
        offset_label.setObjectName("parameterLabel")

        stacked_layout.addWidget(self.check_stacked_auto_offset, 0, 0, 1, 2)
        stacked_layout.addWidget(offset_label, 1, 0)
        stacked_layout.addWidget(self.input_stacked_offset, 1, 1)
        stacked_layout.addWidget(self.check_stacked_labels, 2, 0, 1, 2)

        maximum_label = QLabel(tr("Maximum spectra:"))
        maximum_label.setObjectName("parameterLabel")

        stacked_layout.addWidget(maximum_label, 3, 0)
        stacked_layout.addWidget(self.input_stacked_max_spectra, 3, 1)
        stacked_layout.addWidget(self.check_stacked_by_type, 4, 0, 1, 2)
        stacked_layout.addWidget(self.check_stacked_limited, 5, 0, 1, 2)

        configuration_content_layout.addWidget(self.stacked_group)
        configuration_content_layout.addStretch()

        configuration_scroll.setWidget(configuration_content)
        configuration_layout.addWidget(configuration_scroll)
        visualization_tab_layout.addWidget(configuration_group, 2)

        self.options_tabs.addTab(
            visualization_tab,
            tr("Visualization"),
        )

        # CSV export tab
        export_tab = QWidget()
        export_tab_layout = QHBoxLayout(export_tab)
        export_tab_layout.setContentsMargins(10, 10, 10, 10)
        export_tab_layout.setSpacing(12)

        export_group = QGroupBox(tr("CSV export options"))
        export_layout = QVBoxLayout(export_group)
        export_layout.setContentsMargins(16, 20, 16, 14)
        export_layout.setSpacing(4)

        self.export_group = QButtonGroup(self)
        self.export_group.setExclusive(True)

        self.export_none = QRadioButton(tr("Do not export a CSV file"))
        self.export_full = QRadioButton(tr("Export full matrix as .csv"))
        self.export_limited = QRadioButton(tr("Export limited-range matrix as .csv"))
        self.export_type = QRadioButton(tr("Export matrix by sample type as .csv"))
        self.export_limited_type = QRadioButton(
            tr("Export limited-range matrix by sample type as .csv")
        )

        export_buttons = [
            self.export_none,
            self.export_full,
            self.export_limited,
            self.export_type,
            self.export_limited_type,
        ]

        for index, button in enumerate(export_buttons):
            self.export_group.addButton(button, index)
            export_layout.addWidget(button)

        self.export_none.setChecked(True)
        export_layout.addStretch()
        export_tab_layout.addWidget(export_group, 1)

        export_configuration_group = QGroupBox(tr("Export configuration"))
        export_configuration_layout = QVBoxLayout(export_configuration_group)
        export_configuration_layout.setContentsMargins(10, 18, 10, 10)

        export_configuration_scroll = QScrollArea()
        export_configuration_scroll.setWidgetResizable(True)
        export_configuration_scroll.setFrameShape(QFrame.NoFrame)
        export_configuration_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        export_configuration_content = QWidget()
        export_configuration_content_layout = QVBoxLayout(export_configuration_content)
        export_configuration_content_layout.setContentsMargins(4, 4, 8, 4)
        export_configuration_content_layout.setSpacing(10)

        self.export_configuration_empty = QLabel(
            "Select a CSV export option that requires " "additional parameters."
        )
        self.export_configuration_empty.setWordWrap(True)
        self.export_configuration_empty.setObjectName("parameterLabel")
        export_configuration_content_layout.addWidget(self.export_configuration_empty)

        self.export_range_group = QGroupBox(tr("Export X-axis range"))
        export_range_layout = QGridLayout(self.export_range_group)
        export_range_layout.setContentsMargins(14, 20, 14, 12)
        export_range_layout.setHorizontalSpacing(10)
        export_range_layout.setVerticalSpacing(8)

        export_minimum_label = QLabel(tr("Minimum X:"))
        export_minimum_label.setObjectName("parameterLabel")
        export_maximum_label = QLabel(tr("Maximum X:"))
        export_maximum_label.setObjectName("parameterLabel")

        self.input_export_range_min = QLineEdit()
        self.input_export_range_max = QLineEdit()
        self.input_export_range_min.setPlaceholderText(tr("Minimum value"))
        self.input_export_range_max.setPlaceholderText(tr("Maximum value"))

        export_range_layout.addWidget(export_minimum_label, 0, 0)
        export_range_layout.addWidget(self.input_export_range_min, 0, 1)
        export_range_layout.addWidget(export_maximum_label, 1, 0)
        export_range_layout.addWidget(self.input_export_range_max, 1, 1)
        export_configuration_content_layout.addWidget(self.export_range_group)

        self.export_type_group = QGroupBox(tr("Export sample type"))
        export_type_layout = QVBoxLayout(self.export_type_group)
        export_type_layout.setContentsMargins(14, 20, 14, 12)
        export_type_layout.setSpacing(8)

        export_type_label = QLabel(tr("Choose the sample type to export:"))
        export_type_label.setObjectName("parameterLabel")
        self.combo_export_sample_type = QComboBox()

        export_type_layout.addWidget(export_type_label)
        export_type_layout.addWidget(self.combo_export_sample_type)
        export_configuration_content_layout.addWidget(self.export_type_group)

        self.export_name_group = QGroupBox(tr("CSV file name"))
        export_name_layout = QVBoxLayout(self.export_name_group)
        export_name_layout.setContentsMargins(14, 20, 14, 12)
        export_name_layout.setSpacing(8)

        export_name_label = QLabel(tr("Enter the file name:"))
        export_name_label.setObjectName("parameterLabel")

        self.input_export_file_name = QLineEdit()
        self.input_export_file_name.setPlaceholderText(tr("E.g.: exported_spectra.csv"))

        export_name_help = QLabel(
            tr("The file will be saved in the application's current folder.")
        )
        export_name_help.setObjectName("parameterLabel")
        export_name_help.setWordWrap(True)

        export_name_layout.addWidget(export_name_label)
        export_name_layout.addWidget(self.input_export_file_name)
        export_name_layout.addWidget(export_name_help)

        export_configuration_content_layout.addWidget(self.export_name_group)
        export_configuration_content_layout.addStretch()

        export_configuration_scroll.setWidget(export_configuration_content)
        export_configuration_layout.addWidget(export_configuration_scroll)
        export_tab_layout.addWidget(export_configuration_group, 2)

        self.options_tabs.addTab(export_tab, tr("CSV export"))
        main_layout.addWidget(self.options_tabs, 1)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        back_button = QPushButton(tr("Back"))
        back_button.setObjectName("cancelButton")
        back_button.clicked.connect(self.cancel_requested.emit)

        accept_button = QPushButton(tr("Accept"))
        accept_button.setObjectName("acceptButton")
        accept_button.clicked.connect(self.confirm_selection)

        buttons_layout.addStretch()
        buttons_layout.addWidget(back_button)
        buttons_layout.addWidget(accept_button)
        main_layout.addLayout(buttons_layout)

        for control in (
            self.check_limited_plot,
            self.check_type_plot,
            self.check_limited_type_plot,
            self.check_stacked_plot,
            self.check_stacked_by_type,
            self.check_stacked_limited,
        ):
            control.toggled.connect(self.update_parameter_visibility)

        for export_button in export_buttons:
            export_button.toggled.connect(self.update_parameter_visibility)

        self.update_parameter_visibility()

        # Inicializar parámetros
        self.update_dataset_parameters(self.combo_archivo.currentIndex())

    def update_parameter_visibility(self):
        """Show only parameters required by each active tab."""
        plot_needs_range = (
            self.check_limited_plot.isChecked()
            or self.check_limited_type_plot.isChecked()
            or (
                self.check_stacked_plot.isChecked()
                and self.check_stacked_limited.isChecked()
            )
        )

        plot_needs_type = (
            self.check_type_plot.isChecked()
            or self.check_limited_type_plot.isChecked()
            or (
                self.check_stacked_plot.isChecked()
                and self.check_stacked_by_type.isChecked()
            )
        )

        plot_needs_stacked = self.check_stacked_plot.isChecked()

        self.range_group.setVisible(plot_needs_range)
        self.type_group.setVisible(plot_needs_type)
        self.stacked_group.setVisible(plot_needs_stacked)
        self.configuration_empty.setVisible(
            not (plot_needs_range or plot_needs_type or plot_needs_stacked)
        )

        checked_export = self.export_group.checkedButton()
        export_id = (
            self.export_group.id(checked_export) if checked_export is not None else 0
        )

        export_needs_range = export_id in {2, 4}
        export_needs_type = export_id in {3, 4}

        self.export_range_group.setVisible(export_needs_range)
        self.export_type_group.setVisible(export_needs_type)

        export_enabled = (
            checked_export is not None and self.export_group.id(checked_export) != 0
        )
        self.export_name_group.setVisible(export_enabled)

        self.export_configuration_empty.setVisible(
            not (export_needs_range or export_needs_type or export_enabled)
        )

    def update_dataset_parameters(self, index):
        """
        Updates range limits and sample types when
        the selected dataset changes.
        """

        self.combo_sample_type.clear()
        self.combo_export_sample_type.clear()

        if index < 0 or index >= len(self.dataframes):
            self.input_range_min.clear()
            self.input_range_max.clear()
            self.input_export_range_min.clear()
            self.input_export_range_max.clear()
            return

        df = self.dataframes[index]

        if df is None or df.empty:
            return

        # Tipos de muestra almacenados en la primera fila
        sample_types = df.iloc[0, 1:].dropna().astype(str).tolist()

        unique_types = list(dict.fromkeys(sample_types))

        self.combo_sample_type.addItems(unique_types)
        self.combo_export_sample_type.addItems(unique_types)

        # Valores del eje X
        x_values = pd.to_numeric(
            df.iloc[1:, 0],
            errors="coerce",
        ).dropna()

        if x_values.empty:
            self.input_range_min.clear()
            self.input_range_max.clear()
            self.input_export_range_min.clear()
            self.input_export_range_max.clear()
            return

        self.input_range_min.setText(f"{float(x_values.min()):g}")

        self.input_range_max.setText(f"{float(x_values.max()):g}")
        self.input_export_range_min.setText(f"{float(x_values.min()):g}")
        self.input_export_range_max.setText(f"{float(x_values.max()):g}")

        current_name = os.path.splitext(os.path.basename(self.rutas_completas[index]))[
            0
        ]
        if not self.input_export_file_name.text().strip():
            self.input_export_file_name.setText(f"{current_name}_export.csv")

    def confirm_selection(self):
        if self.combo_archivo.count() == 0:
            QMessageBox.warning(
                self,
                tr("No dataset"),
                tr("No dataset is available."),
            )
            return

        plots = {
            "full": self.check_full_plot.isChecked(),
            "limited": self.check_limited_plot.isChecked(),
            "type": self.check_type_plot.isChecked(),
            "limited_type": (self.check_limited_type_plot.isChecked()),
            "stacked": self.check_stacked_plot.isChecked(),
        }

        selected_export_button = self.export_group.checkedButton()
        selected_export_id = (
            self.export_group.id(selected_export_button)
            if selected_export_button is not None
            else 0
        )

        export_action_map = {
            1: "full",
            2: "limited",
            3: "type",
            4: "limited_type",
        }
        export_action = export_action_map.get(selected_export_id)

        has_plot = any(plots.values())

        if not has_plot and export_action is None:
            QMessageBox.warning(
                self,
                tr("No operation"),
                tr(
                    "Select at least one visualization "
                    "or one CSV export operation."
                ),
            )
            return

        plot_requires_range = (
            plots["limited"]
            or plots["limited_type"]
            or (plots["stacked"] and self.check_stacked_limited.isChecked())
        )

        plot_requires_type = (
            plots["type"]
            or plots["limited_type"]
            or (plots["stacked"] and self.check_stacked_by_type.isChecked())
        )

        export_requires_range = export_action in {
            "limited",
            "limited_type",
        }

        export_requires_type = export_action in {
            "type",
            "limited_type",
        }

        range_min = None
        range_max = None

        if plot_requires_range:
            try:
                range_min = float(self.input_range_min.text().strip())
                range_max = float(self.input_range_max.text().strip())
            except ValueError:
                QMessageBox.warning(
                    self,
                    tr("Invalid plot range"),
                    tr(
                        "Minimum and maximum X values for "
                        "visualization must be numeric."
                    ),
                )
                return

            if range_min >= range_max:
                QMessageBox.warning(
                    self,
                    tr("Invalid plot range"),
                    tr(
                        "Visualization minimum X must be lower "
                        "than maximum X."
                    ),
                )
                return

        sample_type = None

        if plot_requires_type:
            sample_type = self.combo_sample_type.currentText().strip()
            if not sample_type:
                QMessageBox.warning(
                    self,
                    tr("No plot sample type"),
                    tr("Select a sample type for visualization."),
                )
                return

        export_range_min = None
        export_range_max = None

        if export_requires_range:
            try:
                export_range_min = float(self.input_export_range_min.text().strip())
                export_range_max = float(self.input_export_range_max.text().strip())
            except ValueError:
                QMessageBox.warning(
                    self,
                    tr("Invalid export range"),
                    tr(
                        "Minimum and maximum X values for "
                        "CSV export must be numeric."
                    ),
                )
                return

            if export_range_min >= export_range_max:
                QMessageBox.warning(
                    self,
                    tr("Invalid export range"),
                    tr(
                        "Export minimum X must be lower "
                        "than maximum X."
                    ),
                )
                return

        export_file_name = None

        if export_action is not None:
            export_file_name = self.input_export_file_name.text().strip()

            if not export_file_name:
                QMessageBox.warning(
                    self,
                    tr("Invalid file name"),
                    tr("Enter a name for the CSV file."),
                )
                self.input_export_file_name.setFocus()
                return

            invalid_characters = '<>:"/\\|?*'
            if any(character in export_file_name for character in invalid_characters):
                QMessageBox.warning(
                    self,
                    tr("Invalid file name"),
                    tr("The file name contains invalid characters."),
                )
                self.input_export_file_name.setFocus()
                return

            if not export_file_name.lower().endswith(".csv"):
                export_file_name += ".csv"

        export_sample_type = None

        if export_requires_type:
            export_sample_type = self.combo_export_sample_type.currentText().strip()
            if not export_sample_type:
                QMessageBox.warning(
                    self,
                    tr("No export sample type"),
                    tr("Select a sample type for CSV export."),
                )
                return

        stacked_options = None

        if plots["stacked"]:
            try:
                stacked_offset = float(self.input_stacked_offset.text().strip())
            except ValueError:
                QMessageBox.warning(
                    self,
                    tr("Invalid offset"),
                    tr("The stacked-spectrum offset must be numeric."),
                )
                return

            if stacked_offset <= 0:
                QMessageBox.warning(
                    self,
                    tr("Invalid offset"),
                    tr(
                        "The stacked-spectrum offset must be "
                        "greater than zero."
                    ),
                )
                return

            try:
                maximum_spectra = int(self.input_stacked_max_spectra.text().strip())
            except ValueError:
                QMessageBox.warning(
                    self,
                    tr("Invalid maximum"),
                    tr("Maximum spectra must be a whole number."),
                )
                return

            if maximum_spectra <= 0:
                QMessageBox.warning(
                    self,
                    tr("Invalid maximum"),
                    tr("Maximum spectra must be greater than zero."),
                )
                return

            stacked_options = {
                "offset_mode": (
                    "automatic"
                    if self.check_stacked_auto_offset.isChecked()
                    else "manual"
                ),
                "offset_value": stacked_offset,
                "show_labels": (self.check_stacked_labels.isChecked()),
                "maximum_spectra": maximum_spectra,
                "sample_type": (
                    sample_type if self.check_stacked_by_type.isChecked() else None
                ),
                "range_min": (
                    range_min if self.check_stacked_limited.isChecked() else None
                ),
                "range_max": (
                    range_max if self.check_stacked_limited.isChecked() else None
                ),
            }

        configuration = {
            "plots": plots,
            "range_min": range_min,
            "range_max": range_max,
            "sample_type": sample_type,
            "stacked_options": stacked_options,
            "export_action": export_action,
            "export_file_name": export_file_name,
            "export_range_min": export_range_min,
            "export_range_max": export_range_max,
            "export_sample_type": export_sample_type,
        }

        self.seleccion_confirmada.emit(
            self.combo_archivo.currentIndex(),
            configuration,
        )