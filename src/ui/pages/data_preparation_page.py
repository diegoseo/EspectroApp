import os
import re

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QMessageBox,
    QLineEdit,
    QTableView,
    QHeaderView,
    QSplitter,
    QTextEdit,
    QScrollArea,
)

from ui.pages.dataframe_page import PandasTableModel
from core.translations import translate, get_language, retranslate_widget_tree


class DataPreparationAssistant(QWidget):
    def tr(self, text, **values):
        return translate(text, get_language(), **values)

    """Convert heterogeneous tabular files to EspectroApp's internal format."""

    prepared_data = Signal(object, str)
    back_requested = Signal()

    def __init__(self, dataframes, file_names, parent=None):
        super().__init__(parent)
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.dataframes = dataframes
        self.file_names = file_names
        self.preview_df = None
        self.active_df = None
        self._updating_sheet = False
        self._updating_header_rows = False

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        back = QPushButton(self.tr("← Back"))
        back.clicked.connect(self.back_requested.emit)
        top.addWidget(back)
        top.addWidget(QLabel(f"<h2>{self.tr('Data Preparation Assistant')}</h2>"))
        top.addStretch()
        root.addLayout(top)

        self.dataset = QComboBox()
        for index, (df, name) in enumerate(zip(dataframes, file_names)):
            status = str(df.attrs.get("data_status", "ready")).upper()
            self.dataset.addItem(f"[{status}] {os.path.basename(str(name))}", index)

        self.sheet = QComboBox()
        self.sheet.setEnabled(False)

        self.orientation = QComboBox()
        self.orientation.addItem(self.tr("Each sample is in a column"), "columns")
        self.orientation.addItem(self.tr("Each sample is in a row"), "rows")

        self.first_intensity_cell = QLineEdit()
        self.first_intensity_cell.setPlaceholderText(self.tr("E.g.: [2, 2]"))

        self.header_rows = QSpinBox()
        self.header_rows.setRange(1, 20)
        self.header_rows.setValue(1)

        # Controls used when each sample is stored in one column.
        self.axis_col = QSpinBox()
        self.axis_col.setRange(1, 99999)

        # User-facing row selectors use one-based numbering.
        # Value 0 means that the row is not used.
        self.sample_name_row = QSpinBox()
        self.sample_name_row.setMinimum(0)
        self.sample_name_row.setSpecialValueText(self.tr("Not used"))

        self.class_row = QSpinBox()
        self.class_row.setMinimum(0)
        self.class_row.setSpecialValueText(self.tr("Not used"))

        self.first_sample_col = QSpinBox()
        self.first_sample_col.setRange(1, 99999)
        self.first_sample_col.setValue(2)

        # Controls used when each sample is stored in one row.
        self.name_col = QSpinBox()
        self.name_col.setRange(1, 99999)

        # User-facing column selector uses one-based numbering.
        # Value 0 means that the column is not used.
        self.class_col = QSpinBox()
        self.class_col.setMinimum(0)
        self.class_col.setSpecialValueText(self.tr("Not used"))

        self.spectral_start = QSpinBox()
        self.spectral_start.setRange(1, 99999)
        self.spectral_start.setValue(2)

        self.suffix_treatment = QComboBox()
        self.suffix_treatment.addItem(self.tr("Keep full sample names"), "keep")
        self.suffix_treatment.addItem(
            self.tr("Remove duplicate endings (.1, .2, ...)"), "pandas"
        )
        self.suffix_treatment.addItem(
            self.tr("Remove numbers at the end (_1, .1, -1, ...)"), "numeric"
        )

        self.sample_name_source = QComboBox()
        self.sample_name_source.addItem(self.tr("Find next to the intensity matrix"), "adjacent")
        self.sample_name_source.addItem(self.tr("Specify manually"), "explicit")
        self.sample_name_source.addItem(self.tr("Generate sample names"), "generated")

        self.class_source = QComboBox()
        self.class_source.addItem(self.tr("Do not use classes"), "none")
        self.class_source.addItem(self.tr("Find next to the sample names"), "adjacent")
        self.class_source.addItem(self.tr("Specify manually"), "explicit")
        self.class_source.addItem(self.tr("Create classes from sample names"), "derive")
        self.class_source.addItem(self.tr("Use one class for all samples"), "generic")

        self.decimal_separator = QComboBox()
        self.decimal_separator.addItem(self.tr("Detect automatically"), "auto")
        self.decimal_separator.addItem(self.tr("Point: 0.125"), ".")
        self.decimal_separator.addItem(self.tr("Comma: 0,125"), ",")

        self.missing = QComboBox()
        self.missing.addItem(self.tr("Fill empty cells automatically"), "interpolate")
        self.missing.addItem(self.tr("Remove samples with empty cells"), "remove")
        self.missing.addItem(self.tr("Remove incomplete spectral points"), "trim")
        self.missing.addItem(self.tr("Leave empty cells unchanged"), "keep")

        box = QGroupBox(self.tr("Configuration"))
        box_layout = QVBoxLayout(box)

        essential_group = QGroupBox(self.tr("Build the spectral matrix"))
        essential_layout = QVBoxLayout(essential_group)

        dataset_grid = QGridLayout()
        dataset_grid.addWidget(QLabel(self.tr("Dataset") + ":"), 0, 0)
        dataset_grid.addWidget(self.dataset, 0, 1)
        dataset_grid.addWidget(QLabel(self.tr("Excel sheet to use") + ":"), 1, 0)
        dataset_grid.addWidget(self.sheet, 1, 1)
        dataset_grid.addWidget(QLabel(self.tr("Where are the samples?") + ":"), 2, 0)
        dataset_grid.addWidget(self.orientation, 2, 1)
        essential_layout.addLayout(dataset_grid)

        self.selection_instruction = QLabel()
        self.selection_instruction.setWordWrap(True)
        self.selection_instruction.setStyleSheet(
            "QLabel { background:#EEF6F2; border:1px solid #B8D8CA; "
            "border-radius:6px; padding:9px; font-weight:600; }"
        )
        essential_layout.addWidget(self.selection_instruction)

        selection_buttons = QHBoxLayout()
        self.start_selection_button = QPushButton(self.tr("Start guided selection"))
        self.reset_selection_button = QPushButton(self.tr("Reset selection"))
        self.start_selection_button.clicked.connect(self.start_guided_selection)
        self.reset_selection_button.clicked.connect(self.reset_guided_selection)
        selection_buttons.addWidget(self.start_selection_button)
        selection_buttons.addWidget(self.reset_selection_button)
        essential_layout.addLayout(selection_buttons)

        self.selection_result = QLabel()
        self.selection_result.setWordWrap(True)
        essential_layout.addWidget(self.selection_result)

        box_layout.addWidget(essential_group)

        cleaning_group = QGroupBox(self.tr("Cleaning and validation"))
        cleaning_grid = QGridLayout(cleaning_group)
        cleaning_fields = [
            (self.tr("Sample suffixes"), self.suffix_treatment),
            (self.tr("Empty cells"), self.missing),
        ]
        self.cleaning_rows = []
        for row, (text, widget) in enumerate(cleaning_fields):
            label = QLabel(text + ":")
            cleaning_grid.addWidget(label, row, 0)
            cleaning_grid.addWidget(widget, row, 1)
            self.cleaning_rows.append((label, widget))
        box_layout.addWidget(cleaning_group)

        # Compatibility controls remain available internally, but are no longer
        # exposed to the user. They are filled by the guided table selection.
        self.detection_summary = QLabel()
        self.detection_summary.hide()
        self.advanced_box = QGroupBox()
        self.advanced_box.hide()
        self.configuration_rows = []
        self.first_intensity_cell.hide()
        self.sample_name_source.hide()
        self.class_source.setCurrentIndex(self.class_source.findData("derive"))
        self.class_source.hide()
        self.decimal_separator.hide()

        self._selection_step = -1
        self._selected_name_header = None
        self._selected_axis_header = None
        self._selected_intensity_cell = None
        self._selected_class_header = None
        split = QSplitter(Qt.Horizontal)

        # Keep the configuration panel usable on smaller screens.
        # Only the left panel scrolls; the data previews remain visible.
        configuration_scroll = QScrollArea()
        configuration_scroll.setWidgetResizable(True)
        configuration_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        configuration_scroll.setFrameShape(QScrollArea.NoFrame)
        configuration_scroll.setWidget(box)
        configuration_scroll.setMinimumWidth(390)
        split.addWidget(configuration_scroll)

        views = QSplitter(Qt.Vertical)
        self.raw = QTableView()
        self.prep = QTableView()
        for title, view in (
            (self.tr("Raw preview"), self.raw),
            (self.tr("Prepared preview"), self.prep),
        ):
            group = QGroupBox(title)
            layout = QVBoxLayout(group)
            layout.addWidget(view)
            view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            views.addWidget(group)
        split.addWidget(views)
        self.raw.verticalHeader().setSectionsClickable(True)
        self.raw.horizontalHeader().setSectionsClickable(True)
        self.raw.verticalHeader().sectionClicked.connect(self.on_raw_vertical_header_clicked)
        self.raw.horizontalHeader().sectionClicked.connect(self.on_raw_horizontal_header_clicked)
        self.raw.clicked.connect(self.on_raw_cell_clicked)
        split.setSizes([430, 900])
        root.addWidget(split, 1)

        self.selection_info = QLabel()
        self.selection_info.setWordWrap(True)
        root.addWidget(self.selection_info)

        self.status = QLabel(self.tr("Generate a preview to validate the dataset."))
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        validation_box = QGroupBox(self.tr("Validation report"))
        validation_layout = QVBoxLayout(validation_box)
        self.validation_report = QTextEdit()
        self.validation_report.setReadOnly(True)
        self.validation_report.setMaximumHeight(155)
        self.validation_report.setPlaceholderText(
            "Generate a preview to see structural checks and corrections."
        )
        validation_layout.addWidget(self.validation_report)
        root.addWidget(validation_box)

        buttons = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText(self.tr("Output dataset name"))
        self.preview_button = QPushButton(self.tr("Generate preview"))
        self.save_button = QPushButton(self.tr("Save as READY dataset"))
        self.save_button.setEnabled(False)
        self.preview_button.clicked.connect(self.generate_preview)
        self.save_button.clicked.connect(self.save)
        buttons.addWidget(self.name, 1)
        buttons.addWidget(self.preview_button)
        buttons.addWidget(self.save_button)
        root.addLayout(buttons)

        self.dataset.currentIndexChanged.connect(self.load_selected)
        self.sheet.currentIndexChanged.connect(self.load_selected_sheet)
        self.orientation.currentIndexChanged.connect(self.toggle)

        for widget in (
            self.header_rows,
            self.axis_col,
            self.first_sample_col,
            self.name_col,
            self.spectral_start,
            self.sample_name_row,
            self.class_row,
            self.class_col,
        ):
            widget.valueChanged.connect(self.invalidate_preview)

        for widget in (
            self.suffix_treatment,
            self.decimal_separator,
            self.missing,
        ):
            widget.currentIndexChanged.connect(self.invalidate_preview)

        self.load_selected()
        self.reset_guided_selection()


    def start_guided_selection(self):
        """Start the three-click matrix-building workflow."""
        self._selection_step = 0
        self._selected_name_header = None
        self._selected_axis_header = None
        self._selected_intensity_cell = None
        self._selected_class_header = None
        self.prep.setModel(None)
        self.preview_df = None
        self.save_button.setEnabled(False)
        self.update_guided_instruction()

    def reset_guided_selection(self):
        self._selection_step = -1
        self._selected_name_header = None
        self._selected_axis_header = None
        self._selected_intensity_cell = None
        self._selected_class_header = None
        self.class_source.setCurrentIndex(self.class_source.findData("derive"))
        self.selection_instruction.setText(
            self.tr("Press Start guided selection and identify the matrix directly in the raw preview.")
        )
        self.selection_result.setText(self.tr("No matrix structure has been selected yet."))
        self.invalidate_preview()

    def update_guided_instruction(self):
        orientation = self.effective_orientation()
        if self._selection_step == 0:
            target = self.tr("row number") if orientation == "columns" else self.tr("column number")
            self.selection_instruction.setText(
                self.tr("Step 1 of 3: click the {target} containing the sample names.", target=target)
            )
        elif self._selection_step == 1:
            target = self.tr("column number") if orientation == "columns" else self.tr("row number")
            self.selection_instruction.setText(
                self.tr("Step 2 of 3: click the {target} containing the spectral axis.", target=target)
            )
        elif self._selection_step == 2:
            self.selection_instruction.setText(
                self.tr("Step 3 of 3: click the first intensity value in the matrix.")
            )
        else:
            self.selection_instruction.setText(self.tr("The matrix structure is ready for validation."))
        self.update_selection_result()

    def update_selection_result(self):
        parts = []
        if self._selected_name_header is not None:
            parts.append(self.tr("Sample names selected: {value}", value=self._selected_name_header + 1))
        if self._selected_axis_header is not None:
            parts.append(self.tr("Spectral axis selected: {value}", value=self._selected_axis_header + 1))
        if self._selected_intensity_cell is not None:
            row, column = self._selected_intensity_cell
            parts.append(self.tr("First intensity selected: [{row}, {column}]", row=row + 1, column=column + 1))
        self.selection_result.setText(" · ".join(parts) if parts else self.tr("No matrix structure has been selected yet."))

    def on_raw_vertical_header_clicked(self, section):
        orientation = self.effective_orientation()
        if self._selection_step == 0 and orientation == "columns":
            self._selected_name_header = section
            self.sample_name_row.setValue(section + 1)
            self.sample_name_source.setCurrentIndex(self.sample_name_source.findData("explicit"))
            self._selection_step = 1
        elif self._selection_step == 1 and orientation == "rows":
            self._selected_axis_header = section
            self.header_rows.setValue(section + 1)
            self._selection_step = 2
        else:
            return
        self.update_guided_instruction()

    def on_raw_horizontal_header_clicked(self, section):
        orientation = self.effective_orientation()
        if self._selection_step == 0 and orientation == "rows":
            self._selected_name_header = section
            self.name_col.setValue(section + 1)
            self.sample_name_source.setCurrentIndex(self.sample_name_source.findData("explicit"))
            self._selection_step = 1
        elif self._selection_step == 1 and orientation == "columns":
            self._selected_axis_header = section
            self.axis_col.setValue(section + 1)
            self._selection_step = 2
        else:
            return
        self.update_guided_instruction()

    def on_raw_cell_clicked(self, model_index):
        if self._selection_step != 2 or not model_index.isValid():
            return
        row, column = model_index.row(), model_index.column()
        self._selected_intensity_cell = (row, column)
        self.first_intensity_cell.setText(self.format_cell_reference(row, column))
        if self.effective_orientation() == "columns":
            if self._selected_axis_header is None or column <= self._selected_axis_header:
                self.status.setText(self.tr("The first intensity must be to the right of the spectral axis."))
                return
            self.header_rows.setValue(row)
            self.first_sample_col.setValue(column + 1)
        else:
            if self._selected_axis_header is None or row <= self._selected_axis_header:
                self.status.setText(self.tr("The first intensity must be below the spectral axis."))
                return
            self.header_rows.setValue(row)
            self.spectral_start.setValue(column + 1)
        self._selection_step = -1
        self.update_guided_instruction()
        self.generate_preview()

    def on_structure_source_changed(self, *_):
        """Reveal exact-location fields only when the user requests them."""
        needs_manual_location = self.sample_name_source.currentData() == "explicit"
        if needs_manual_location and not self.advanced_box.isChecked():
            self.advanced_box.setChecked(True)
        self.update_advanced_visibility()
        self.invalidate_preview()

    @staticmethod
    def column_letters_to_index(letters):
        value = 0
        for character in letters.upper():
            if not ("A" <= character <= "Z"):
                raise ValueError("Invalid column letters.")
            value = value * 26 + (ord(character) - ord("A") + 1)
        return value - 1

    @staticmethod
    def index_to_column_letters(index):
        index = int(index) + 1
        letters = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            letters = chr(ord("A") + remainder) + letters
        return letters

    def parse_cell_reference(self, text):
        """Parse a user coordinate written as [row, column], using 1-based values."""
        match = re.fullmatch(
            r"\s*\[?\s*(\d+)\s*[,;]\s*(\d+)\s*\]?\s*",
            str(text or ""),
        )
        if not match:
            raise ValueError(
                self.tr("Use a coordinate in the format [row, column], for example [2, 2].")
            )
        row = int(match.group(1)) - 1
        column = int(match.group(2)) - 1
        if row < 0 or column < 0:
            raise ValueError(self.tr("Row and column numbers must start at 1."))
        return row, column

    def format_cell_reference(self, row, column):
        return f"[{int(row) + 1}, {int(column) + 1}]"

    def apply_first_intensity_cell(self):
        """Convert the manually entered Excel cell into matrix positions."""
        if self.active_df is None:
            return
        try:
            row, column = self.parse_cell_reference(self.first_intensity_cell.text())
            if row >= self.active_df.shape[0] or column >= self.active_df.shape[1]:
                raise ValueError(self.tr("The selected cell is outside the dataset."))
            self.header_rows.blockSignals(True)
            self.header_rows.setValue(max(1, row))
            self.header_rows.blockSignals(False)
            if self.effective_orientation() == "columns":
                if column <= 0:
                    raise ValueError(self.tr("The first intensity must have the spectral axis immediately to its left."))
                self.axis_col.setValue(column)
                self.first_sample_col.setValue(column + 1)
            else:
                if row <= 0:
                    raise ValueError(self.tr("The first intensity must have the spectral axis immediately above it."))
                self.spectral_start.setValue(column + 1)
            self.update_detection_summary()
            self.invalidate_preview()
        except Exception as error:
            self.status.setText(str(error))

    @staticmethod
    def clean(value, fallback):
        if pd.isna(value):
            return fallback
        text = re.sub(r"\s+", " ", str(value)).strip()
        return text or fallback

    def clean_sample_names(self, values):
        """Clean visible sample names while preserving repeated names."""
        mode = self.suffix_treatment.currentData()
        cleaned = []
        changed = 0
        for index, value in enumerate(values, start=1):
            name = self.clean(value, f"Sample {index}")
            original = name
            if mode == "pandas":
                name = re.sub(r"\.\d+$", "", name).strip()
            elif mode == "numeric":
                name = re.sub(r"(?:[_\.\-]\s*\d+)$", "", name).strip()
            if not name:
                name = f"Sample {index}"
            if name != original:
                changed += 1
            cleaned.append(name)
        return cleaned, changed

    @staticmethod
    def make_internal_sample_keys(values):
        """Create unique internal keys without changing visible names."""
        counts = {}
        keys = []
        for index, value in enumerate(values, start=1):
            base = re.sub(r"\s+", " ", str(value)).strip() or f"Sample {index}"
            counts[base] = counts.get(base, 0) + 1
            keys.append(f"{base}::{counts[base]}")
        return keys

    def detect_decimal_separator(self):
        """Infer whether decimal values use a point or comma.

        Detection inspects textual numeric-looking cells and does not confuse
        the column delimiter with the decimal separator.
        """
        df = self.active_df
        if df is None or df.empty:
            return "."

        comma_count = 0
        point_count = 0
        # A representative slice is enough and keeps the preview responsive.
        for value in df.iloc[:200, :80].to_numpy().ravel():
            if pd.isna(value):
                continue
            text = str(value).strip()
            if re.fullmatch(r"[+-]?(?:\d+),(?:\d+)(?:[eE][+-]?\d+)?", text):
                comma_count += 1
            elif re.fullmatch(r"[+-]?(?:\d+)\.(?:\d+)(?:[eE][+-]?\d+)?", text):
                point_count += 1

        return "," if comma_count > point_count else "."

    def effective_decimal_separator(self):
        return self.decimal_separator.currentData() or "."

    def to_numeric_series(self, values):
        """Convert values using the selected/detected decimal separator."""
        series = pd.Series(values, copy=True)
        decimal = self.effective_decimal_separator()

        # Preserve real numeric values; only normalize textual cells.
        def normalize(value):
            if pd.isna(value) or isinstance(value, (int, float, np.number)):
                return value
            text = str(value).strip().replace("\u00a0", "")
            if decimal == ",":
                # Decimal comma files commonly use a point as thousands mark.
                if re.fullmatch(r"[+-]?[\d.]+,\d+(?:[eE][+-]?\d+)?", text):
                    text = text.replace(".", "").replace(",", ".")
                else:
                    text = text.replace(",", ".")
            return text

        return pd.to_numeric(series.map(normalize), errors="coerce")

    def to_numeric_frame(self, frame):
        return frame.apply(lambda column: self.to_numeric_series(column).to_numpy())

    def invalidate_preview(self, *_):
        """Discard any preview generated with an older configuration."""
        self.preview_df = None
        self.prep.setModel(None)
        self.save_button.setEnabled(False)
        self.status.setText(self.tr(self.tr("Configuration changed. Generate a new preview.")))
        self.validation_report.clear()

    def selected_source_dataframe(self):
        data = self.dataset.currentData()
        if data is None:
            raise ValueError("No dataset is selected.")
        index = int(data)
        if index < 0 or index >= len(self.dataframes):
            raise IndexError("The selected dataset is no longer available.")
        return index, self.dataframes[index], self.file_names[index]

    def load_selected(self, *_):
        if not self.dataframes:
            return
        index, source_df, source_name = self.selected_source_dataframe()
        self.preview_df = None
        self.prep.setModel(None)

        available_sheets = list(source_df.attrs.get("available_sheets", []))
        current_sheet = source_df.attrs.get("sheet_name")
        self._updating_sheet = True
        self.sheet.clear()
        if available_sheets:
            self.sheet.addItems([str(value) for value in available_sheets])
            self.sheet.setEnabled(True)
            if current_sheet in available_sheets:
                self.sheet.setCurrentText(str(current_sheet))
        else:
            self.sheet.addItem(self.tr(self.tr("Not applicable")))
            self.sheet.setEnabled(False)
        self._updating_sheet = False

        self.active_df = source_df.copy()
        self.active_df.attrs = source_df.attrs.copy()
        self.update_raw_view(index, source_name)

    def load_selected_sheet(self, *_):
        if self._updating_sheet or not self.sheet.isEnabled():
            return
        try:
            index, source_df, source_name = self.selected_source_dataframe()
            source_path = source_df.attrs.get("source_path")
            sheet_name = self.sheet.currentText()
            if not source_path or not os.path.isfile(source_path):
                raise FileNotFoundError(
                    "The original Excel workbook could not be found."
                )
            dataframe = pd.read_excel(
                source_path, sheet_name=sheet_name, header=None, dtype=object
            )
            dataframe.attrs = source_df.attrs.copy()
            dataframe.attrs["sheet_name"] = sheet_name
            self.active_df = dataframe
            self.invalidate_preview()
            self.update_raw_view(index, source_name)
        except Exception as error:
            QMessageBox.critical(self, self.tr("Worksheet error"), str(error))

    def update_raw_view(self, index, source_name):
        df = self.active_df
        if df is None:
            return

        raw_preview = df.iloc[:200, :80].copy()
        raw_preview.columns = range(1, raw_preview.shape[1] + 1)
        self.raw.setModel(PandasTableModel(raw_preview, self.raw))
        base = os.path.splitext(os.path.basename(str(source_name)))[0]
        sheet_suffix = (
            "_" + self.clean(self.sheet.currentText(), "sheet")
            if self.sheet.isEnabled()
            else ""
        )
        self.name.setText(base + sheet_suffix + "_prepared")

        sheet_text = (
            self.sheet.currentText()
            if self.sheet.isEnabled()
            else self.tr(self.tr("Not applicable"))
        )
        delimiter = df.attrs.get("detected_delimiter")
        delimiter_text = (
            self.delimiter_name(delimiter)
            if delimiter is not None
            else self.tr(self.tr("Not applicable"))
        )
        self.selection_info.setText(
            self.tr(
                "Selected dataset #{number}: {name} · worksheet: {worksheet} · delimiter: {delimiter} · {rows} rows × {columns} columns",
                number=index + 1,
                name=os.path.basename(str(source_name)),
                worksheet=sheet_text,
                delimiter=delimiter_text,
                rows=f"{df.shape[0]:,}",
                columns=f"{df.shape[1]:,}",
            )
        )

        maximum = max(df.shape[1], 1)
        for widget in (
            self.axis_col,
            self.first_sample_col,
            self.name_col,
            self.spectral_start,
        ):
            widget.setMaximum(maximum)

        self.class_col.blockSignals(True)
        self.class_col.setMaximum(df.shape[1])
        if self.class_col.value() > df.shape[1]:
            self.class_col.setValue(0)
        self.class_col.blockSignals(False)

        # Manual-only assistant: use neutral defaults and never scan the
        # dataset to choose its structure.
        self.orientation.blockSignals(True)
        self.orientation.setCurrentIndex(0)
        self.orientation.blockSignals(False)
        self.first_intensity_cell.clear()
        self.sample_name_row.setValue(1)
        self.class_row.setValue(0)
        self.reset_guided_selection()
        self.update_detection_summary()

    @staticmethod
    def delimiter_name(delimiter):
        names = {
            ",": "comma (,)",
            ";": "semicolon (;)",
            "\t": "tab",
            "|": "pipe (|)",
            " ": "whitespace",
        }
        return names.get(delimiter, repr(delimiter) if delimiter else "not detected")

    def decimal_name(self, separator):
        return self.tr("comma (0,125)") if separator == "," else self.tr("point (0.125)")

    def detect_header_rows(self, orientation=None):
        """Detect how many leading rows precede the numeric spectral matrix.

        The returned value is the index of the first data row, which is also
        the number of header rows. Detection is conservative: a candidate row
        must contain a numeric spectral coordinate/intensity block and the
        following row should normally show the same pattern.
        """
        df = self.active_df
        if df is None or df.empty:
            return 1

        orientation = orientation or self.effective_orientation()
        row_limit = min(len(df), 50)

        def numeric_share(values):
            series = pd.Series(list(values), dtype=object)
            if series.empty:
                return 0.0
            return float(self.to_numeric_series(series).notna().mean())

        candidates = []

        if orientation == "columns":
            axis_column = min(max(self.axis_col.value() - 1, 0), df.shape[1] - 1)
            first_sample = min(
                max(self.first_sample_col.value() - 1, 0),
                max(df.shape[1] - 1, 0),
            )
            sample_columns = [
                column
                for column in range(first_sample, df.shape[1])
                if column != axis_column
            ]
            if not sample_columns:
                sample_columns = [
                    column for column in range(df.shape[1]) if column != axis_column
                ]

            for row_index in range(row_limit):
                axis_value = self.to_numeric_series(
                    pd.Series([df.iat[row_index, axis_column]])
                ).iloc[0]
                axis_is_numeric = pd.notna(axis_value)
                share = numeric_share(df.iloc[row_index, sample_columns])
                if axis_is_numeric and share >= 0.60:
                    candidates.append(row_index)

        else:  # samples in rows
            first_spectral = min(
                max(self.spectral_start.value() - 1, 0),
                max(df.shape[1] - 1, 0),
            )
            spectral_columns = list(range(first_spectral, df.shape[1]))

            for row_index in range(row_limit):
                share = numeric_share(df.iloc[row_index, spectral_columns])
                # Require at least two numeric spectral cells where possible.
                numeric_count = int(
                    self.to_numeric_series(
                        pd.Series(df.iloc[row_index, spectral_columns].tolist())
                    )
                    .notna()
                    .sum()
                )
                required = 1 if len(spectral_columns) == 1 else 2
                if share >= 0.60 and numeric_count >= required:
                    candidates.append(row_index)

        if not candidates:
            return 1

        # Prefer the first candidate followed by another data-like row. This
        # avoids treating an isolated numeric metadata row as the matrix start.
        for candidate in candidates:
            if candidate + 1 in candidates:
                return max(1, candidate)

        return max(1, candidates[0])

    def apply_detected_header_rows(self):
        """Update Header rows from the current table without recursive signals."""
        if self.active_df is None or self.active_df.empty:
            return
        detected = self.detect_header_rows()
        detected = min(max(int(detected), 1), min(20, max(len(self.active_df) - 1, 1)))
        self.header_rows.blockSignals(True)
        try:
            self.header_rows.setValue(detected)
        finally:
            self.header_rows.blockSignals(False)
        self.update_header_row_options()

    def update_header_row_options(self, *_):
        """Update simple one-based row selectors.

        The user enters the row number exactly as it appears in the raw preview.
        Row 0 means that the corresponding information is not present.
        """
        if self._updating_header_rows or self.active_df is None:
            return

        self._updating_header_rows = True
        try:
            maximum = len(self.active_df)
            header_count = min(self.header_rows.value(), maximum)

            self.sample_name_row.blockSignals(True)
            self.class_row.blockSignals(True)
            self.sample_name_row.setMaximum(maximum)
            self.class_row.setMaximum(maximum)

            if self.sample_name_row.value() > maximum:
                self.sample_name_row.setValue(0)
            if self.class_row.value() > maximum:
                self.class_row.setValue(0)

            # Helpful automatic defaults. Values shown to the user are one-based.
            if self.effective_orientation() == "columns":
                if header_count >= 2 and self.sample_name_row.value() == 0 and self.class_row.value() == 0:
                    self.sample_name_row.setValue(max(1, header_count - 1))
                    self.class_row.setValue(header_count)
                elif header_count == 1 and self.class_row.value() == 0:
                    self.class_row.setValue(1)
        finally:
            self.sample_name_row.blockSignals(False)
            self.class_row.blockSignals(False)
            self._updating_header_rows = False

        self.invalidate_preview()

    @staticmethod
    def required_position(widget):
        """Convert a required one-based selector to a zero-based index."""
        return max(int(widget.value()) - 1, 0)

    @staticmethod
    def optional_position(widget):
        """Convert a one-based selector to a zero-based index or None."""
        value = int(widget.value())
        return None if value <= 0 else value - 1

    def detected_orientation(self):
        """Infer the most likely sample orientation from the active table."""
        if self.active_df is None or self.active_df.empty:
            return "columns"
        first_row_numeric = (
            self.to_numeric_series(self.active_df.iloc[0]).notna().mean()
        )
        return "rows" if first_row_numeric > 0.55 else "columns"

    def effective_orientation(self):
        return self.orientation.currentData() or "columns"

    def update_detection_summary(self, detection=None):
        """Compatibility hook; the redundant structure summary is hidden."""
        return

    def update_advanced_visibility(self, *_):
        expanded = self.advanced_box.isChecked()
        mode = self.effective_orientation()
        explicit_names = self.sample_name_source.currentData() == "explicit"
        explicit_classes = self.class_source.currentData() == "explicit"

        for label, widget, row_mode in self.configuration_rows:
            relevant_orientation = row_mode == mode
            visible = expanded and relevant_orientation

            if widget in (self.sample_name_row, self.name_col):
                visible = visible and explicit_names
            elif widget in (self.class_row, self.class_col):
                visible = visible and explicit_classes

            label.setVisible(visible)
            widget.setVisible(visible)

        # Suffix cleaning is relevant only when class labels are used.
        for label, widget in self.cleaning_rows:
            label.setVisible(True)
            widget.setVisible(True)

        # Axis/start fields are derived from the first intensity cell, but they
        # remain available when the user opens Advanced options.
        self.advanced_box.setFlat(not expanded)

    def toggle(self, *_):
        self.reset_guided_selection()
        self.update_advanced_visibility()

    def treat(self, x, matrix, labels, identifiers=None):
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        mode = self.missing.currentData()

        if mode == "interpolate":
            matrix = matrix.interpolate(axis=0, limit_direction="both")
        elif mode == "remove":
            keep = ~matrix.isna().any(axis=0)
            matrix = matrix.loc[:, keep]
            labels = [value for value, valid in zip(labels, keep.tolist()) if valid]
            if identifiers is not None:
                identifiers = [
                    value for value, valid in zip(identifiers, keep.tolist()) if valid
                ]
        elif mode == "trim":
            keep = ~matrix.isna().any(axis=1)
            x = x.loc[keep].reset_index(drop=True)
            matrix = matrix.loc[keep].reset_index(drop=True)

        return x, matrix, labels, identifiers

    @staticmethod
    def _text_quality(values):
        texts = [str(v).strip() for v in values if not pd.isna(v) and str(v).strip()]
        if not texts:
            return 0.0, 0.0
        unique_ratio = len(set(texts)) / len(texts)
        text_ratio = sum(pd.to_numeric(pd.Series([t]), errors="coerce").isna().iloc[0] for t in texts) / len(texts)
        return unique_ratio, text_ratio

    def infer_adjacent_name_position(self, df, header_rows, sample_start):
        """Infer only the nearby identifier row/column requested by the user."""
        if self.effective_orientation() == "columns":
            candidates = []
            for row in range(max(0, header_rows)):
                values = df.iloc[row, sample_start:]
                unique_ratio, text_ratio = self._text_quality(values)
                score = unique_ratio * 2.0 + text_ratio + row * 0.001
                candidates.append((score, row))
            return max(candidates)[1] if candidates else None

        candidates = []
        rows = df.iloc[header_rows:]
        for column in range(max(0, sample_start)):
            unique_ratio, text_ratio = self._text_quality(rows.iloc[:, column])
            score = unique_ratio * 2.0 + text_ratio + column * 0.001
            candidates.append((score, column))
        return max(candidates)[1] if candidates else None

    def infer_adjacent_class_position(self, df, header_rows, sample_start, name_position):
        """Find a nearby repeated-text row/column, excluding sample names."""
        if self.effective_orientation() == "columns":
            candidates = []
            for row in range(max(0, header_rows)):
                if row == name_position:
                    continue
                values = [str(v).strip() for v in df.iloc[row, sample_start:] if not pd.isna(v) and str(v).strip()]
                if not values:
                    continue
                uniqueness = len(set(values)) / len(values)
                numeric_share = pd.to_numeric(pd.Series(values), errors="coerce").notna().mean()
                score = (1.0 - uniqueness) * 2.0 + (1.0 - numeric_share) + row * 0.001
                candidates.append((score, row))
            return max(candidates)[1] if candidates else None

        candidates = []
        rows = df.iloc[header_rows:]
        for column in range(max(0, sample_start)):
            if column == name_position:
                continue
            values = [str(v).strip() for v in rows.iloc[:, column] if not pd.isna(v) and str(v).strip()]
            if not values:
                continue
            uniqueness = len(set(values)) / len(values)
            numeric_share = pd.to_numeric(pd.Series(values), errors="coerce").notna().mean()
            score = (1.0 - uniqueness) * 2.0 + (1.0 - numeric_share) + column * 0.001
            candidates.append((score, column))
        return max(candidates)[1] if candidates else None

    def generate_preview(self):
        try:
            index, _, _ = self.selected_source_dataframe()
            if self.active_df is None:
                raise ValueError("The selected dataset has not been loaded.")

            df = (
                self.active_df.copy()
                .dropna(how="all")
                .dropna(axis=1, how="all")
                .reset_index(drop=True)
            )
            if df.empty or df.shape[1] < 2:
                raise ValueError("The source table does not contain enough data.")

            header_rows = self.header_rows.value()
            if header_rows >= len(df):
                raise ValueError("Header rows must leave at least one data row.")

            report = []
            suffix_changes = 0
            raw_missing = 0
            raw_infinite = 0
            duplicated_identifiers = 0

            if self.effective_orientation() == "columns":
                axis_column = self.axis_col.value() - 1
                first_sample = self.first_sample_col.value() - 1
                name_mode = self.sample_name_source.currentData()
                if name_mode == "generated":
                    name_row = None
                elif name_mode == "explicit":
                    name_row = self.optional_position(self.sample_name_row)
                else:
                    name_row = self.infer_adjacent_name_position(df, header_rows, first_sample)

                class_mode = self.class_source.currentData()
                if class_mode == "explicit":
                    class_row = self.optional_position(self.class_row)
                elif class_mode == "adjacent":
                    class_row = self.infer_adjacent_class_position(
                        df, header_rows, first_sample, name_row
                    )
                else:
                    class_row = None

                if axis_column >= df.shape[1]:
                    raise ValueError(
                        "The selected spectral-axis column does not exist."
                    )
                if first_sample >= df.shape[1]:
                    raise ValueError("The first sample column does not exist.")
                if axis_column >= first_sample:
                    raise ValueError(
                        "Spectral-axis column must be before the first sample column."
                    )
                for title, row in (
                    (self.tr("Sample-name row"), name_row),
                    (self.tr("Class row"), class_row),
                ):
                    if row is not None and int(row) >= header_rows:
                        raise ValueError(f"{title} must be before the first spectral data row.")

                sample_columns = list(range(first_sample, df.shape[1]))
                if axis_column in sample_columns:
                    sample_columns.remove(axis_column)
                if len(sample_columns) < 2:
                    raise ValueError("At least two sample columns are required.")

                if name_row is None:
                    raw_identifiers = [
                        f"Sample {j + 1}" for j in range(len(sample_columns))
                    ]
                    report.append(
                        self.tr(
                            "ℹ No sample-name row: identifiers were generated automatically."
                        )
                    )
                else:
                    raw_identifiers = [
                        self.clean(df.iloc[int(name_row), c], f"Sample {j + 1}")
                        for j, c in enumerate(sample_columns)
                    ]

                duplicated_identifiers = len(raw_identifiers) - len(
                    set(raw_identifiers)
                )
                identifiers, suffix_changes = self.clean_sample_names(raw_identifiers)

                class_mode = "derive"
                if class_mode == "adjacent" and class_row is None:
                    raise ValueError(self.tr("No nearby class row could be identified. Select Specify manually or Do not use classes."))
                if class_mode == "none":
                    raw_labels = ["Sample"] * len(raw_identifiers)
                    report.append(self.tr("ℹ Classes were not used; a neutral internal label was assigned."))
                elif class_mode == "generic":
                    raw_labels = ["Unknown"] * len(raw_identifiers)
                    report.append("ℹ One generic class was assigned to all samples.")
                elif class_mode == "derive":
                    raw_labels = list(raw_identifiers)
                    report.append(self.tr("ℹ Sample names are used as analysis labels."))
                elif class_row is None:
                    raw_labels = list(raw_identifiers)
                    report.append("ℹ No class row: sample names are used as labels.")
                else:
                    raw_labels = [
                        self.clean(df.iloc[int(class_row), c], raw_identifiers[j])
                        for j, c in enumerate(sample_columns)
                    ]

                labels = list(identifiers)

                x = self.to_numeric_series(
                    df.iloc[header_rows:, axis_column]
                ).reset_index(drop=True)
                matrix = self.to_numeric_frame(
                    df.iloc[header_rows:, sample_columns]
                ).reset_index(drop=True)

                valid_x = x.notna()
                invalid_x = int((~valid_x).sum())
                x = x.loc[valid_x].reset_index(drop=True)
                matrix = matrix.loc[valid_x].reset_index(drop=True)

            else:
                start = self.spectral_start.value() - 1
                name_mode = self.sample_name_source.currentData()
                if name_mode == "generated":
                    name_column = None
                elif name_mode == "explicit":
                    name_column = self.name_col.value() - 1
                else:
                    name_column = self.infer_adjacent_name_position(df, header_rows, start)

                class_mode = self.class_source.currentData()
                if class_mode == "explicit":
                    class_column = self.optional_position(self.class_col)
                elif class_mode == "adjacent":
                    class_column = self.infer_adjacent_class_position(
                        df, header_rows, start, name_column
                    )
                else:
                    class_column = None

                if start >= df.shape[1]:
                    raise ValueError("The first spectral column does not exist.")
                if name_column is not None and name_column >= df.shape[1]:
                    raise ValueError("The sample-name column does not exist.")
                if name_column is not None and name_column >= start:
                    raise ValueError(
                        "Sample-name column must be before the first spectral column."
                    )
                if class_column is not None and int(class_column) >= start:
                    raise ValueError(
                        "Class column must be before the first spectral column."
                    )

                header_index = header_rows - 1
                raw_x = self.to_numeric_series(
                    df.iloc[header_index, start:]
                ).reset_index(drop=True)
                valid_x = raw_x.notna()
                invalid_x = int((~valid_x).sum())
                x = raw_x.loc[valid_x].reset_index(drop=True)

                rows = df.iloc[header_rows:].reset_index(drop=True)
                if name_column is None:
                    raw_identifiers = [f"Sample {j + 1}" for j in range(len(rows))]
                    report.append(self.tr("ℹ Sample identifiers were generated automatically."))
                else:
                    raw_identifiers = [
                        self.clean(value, f"Sample {j + 1}")
                        for j, value in enumerate(rows.iloc[:, name_column])
                    ]
                duplicated_identifiers = len(raw_identifiers) - len(
                    set(raw_identifiers)
                )
                identifiers, suffix_changes = self.clean_sample_names(raw_identifiers)

                class_mode = "derive"
                if class_mode == "adjacent" and class_column is None:
                    raise ValueError(self.tr("No nearby class column could be identified. Select Specify manually or Do not use classes."))
                if class_mode == "none":
                    raw_labels = ["Sample"] * len(raw_identifiers)
                    report.append(self.tr("ℹ Classes were not used; a neutral internal label was assigned."))
                elif class_mode == "generic":
                    raw_labels = ["Unknown"] * len(raw_identifiers)
                    report.append("ℹ One generic class was assigned to all samples.")
                elif class_mode == "derive":
                    raw_labels = list(raw_identifiers)
                    report.append(self.tr("ℹ Sample names are used as analysis labels."))
                elif class_column is None:
                    raw_labels = list(raw_identifiers)
                    report.append("ℹ No class column: sample names are used as labels.")
                else:
                    raw_labels = [
                        self.clean(value, raw_identifiers[j])
                        for j, value in enumerate(rows.iloc[:, int(class_column)])
                    ]
                labels = list(identifiers)

                matrix = (
                    self.to_numeric_frame(rows.iloc[:, start:])
                    .loc[:, valid_x.to_numpy()]
                    .T.reset_index(drop=True)
                )

            if x.empty:
                raise ValueError("No numeric spectral-axis values were found.")
            if matrix.shape[1] < 2:
                raise ValueError("At least two valid samples are required.")
            if matrix.shape[0] != len(x):
                raise ValueError(
                    "The spectral axis and intensity matrix have different lengths."
                )
            if x.duplicated().any():
                duplicated = int(x.duplicated().sum())
                raise ValueError(
                    f"The spectral axis contains {duplicated} duplicated value(s)."
                )

            raw_missing = int(matrix.isna().sum().sum())
            raw_infinite = int(np.isinf(matrix.to_numpy(dtype=float)).sum())

            x, matrix, labels, identifiers = self.treat(x, matrix, labels, identifiers)

            if matrix.shape[1] < 2:
                raise ValueError(
                    "Fewer than two samples remain after missing-data treatment."
                )
            if matrix.shape[0] == 0:
                raise ValueError(
                    "No spectral points remain after missing-data treatment."
                )

            missing = int(matrix.isna().sum().sum())
            output = pd.DataFrame([["X Axis", *identifiers]])
            body = pd.concat([x.rename(0), matrix], axis=1)
            body.columns = range(body.shape[1])
            output = pd.concat([output, body], ignore_index=True)
            output.columns = range(output.shape[1])

            delimiter = self.active_df.attrs.get("detected_delimiter")
            output.attrs = {
                "data_status": "ready" if missing == 0 else "preview_with_missing",
                # Visible sample names may repeat because replicates are valid.
                # Class membership remains separate metadata for PCA coloring.
                "sample_ids": list(identifiers or self.clean_sample_names(labels)),
                "sample_keys": self.make_internal_sample_keys(identifiers or labels),
                "class_labels": list(labels),
                "source_dataset_index": index,
                "source_path": self.active_df.attrs.get("source_path"),
                "sheet_name": self.active_df.attrs.get("sheet_name"),
                "orientation": self.effective_orientation(),
                "detected_delimiter": delimiter,
                "decimal_separator": self.effective_decimal_separator(),
                "suffix_treatment": self.suffix_treatment.currentData(),
            }

            direction = (
                "ascending"
                if x.is_monotonic_increasing
                else ("descending" if x.is_monotonic_decreasing else "not ordered")
            )
            report.insert(
                0,
                self.tr(
                    "✓ Numeric spectral axis: {points} valid points.",
                    points=f"{len(x):,}",
                ),
            )
            report.append(
                self.tr(
                    "✓ Samples detected: {samples}.", samples=f"{matrix.shape[1]:,}"
                )
            )
            report.append(self.tr("✓ All retained spectra have the same length."))
            report.append(self.tr("✓ No duplicated spectral-axis values."))
            if direction == "not ordered":
                report.append(
                    self.tr("⚠ Spectral axis is not monotonic; consider sorting it.")
                )
            else:
                report.append(
                    self.tr(
                        "✓ Spectral axis is ordered in {direction} direction.",
                        direction=self.tr(direction),
                    )
                )
            if invalid_x:
                report.append(
                    self.tr(
                        "⚠ {count} non-numeric spectral-axis value(s) were ignored.",
                        count=invalid_x,
                    )
                )
            if raw_missing:
                report.append(
                    self.tr(
                        "ℹ {count} missing intensity value(s) existed before treatment.",
                        count=f"{raw_missing:,}",
                    )
                )
            else:
                report.append(self.tr("✓ No missing intensity values were detected."))
            if raw_infinite:
                report.append(
                    self.tr(
                        "ℹ {count} infinite value(s) were treated as missing.",
                        count=f"{raw_infinite:,}",
                    )
                )
            else:
                report.append(self.tr("✓ No infinite intensity values were detected."))
            if duplicated_identifiers:
                report.append(
                    self.tr(
                        "ℹ {count} repeated sample name(s) were preserved.",
                        count=f"{duplicated_identifiers:,}",
                    )
                )
            else:
                report.append(self.tr("✓ No repeated sample names were detected."))
            if suffix_changes:
                report.append(
                    self.tr(
                        "ℹ Numeric suffixes were removed from {count} sample name(s).",
                        count=f"{suffix_changes:,}",
                    )
                )
            else:
                report.append(self.tr("✓ Sample names required no suffix correction."))
            if delimiter is not None:
                report.append(
                    self.tr(
                        "ℹ Detected text delimiter: {delimiter}.",
                        delimiter=self.delimiter_name(delimiter),
                    )
                )
            report.append(
                self.tr(
                    "ℹ Decimal separator used: {separator}.",
                    separator=self.decimal_name(self.effective_decimal_separator()),
                )
            )

            ready = missing == 0
            report.append(
                "\n" + self.tr("READY — dataset can be saved.")
                if ready
                else "\n"
                + self.tr(
                    "NOT READY — {count} missing value(s) remain.", count=f"{missing:,}"
                )
            )
            output.attrs["validation_report"] = list(report)

            self.preview_df = output

            # Build a clean user-facing matrix. Sample identifiers are shown as
            # columns and class labels remain only in DataFrame.attrs/internal
            # legacy row, so the preview contains spectral coordinates and
            # intensities only.
            preview_display = body.copy()
            preview_display.columns = [self.tr("X Axis"), *identifiers]
            self.prep.setModel(
                PandasTableModel(preview_display.iloc[:200, :80], self.prep)
            )
            self.save_button.setEnabled(ready)
            self.status.setText(
                self.tr(
                    "{points} spectral points · {samples} samples · {missing} missing values · {state} · previewing {rows} rows × {columns} columns",
                    points=f"{len(x):,}",
                    samples=f"{matrix.shape[1]:,}",
                    missing=f"{missing:,}",
                    state="READY" if ready else self.tr("Not ready"),
                    rows=f"{min(len(output), 200):,}",
                    columns=f"{min(output.shape[1], 80):,}",
                )
            )
            self.validation_report.setPlainText("\n".join(report))

        except Exception as error:
            self.preview_df = None
            self.prep.setModel(None)
            self.save_button.setEnabled(False)
            self.status.setText(self.tr("Preview generation failed."))
            self.validation_report.setPlainText(f"✗ {error}")
            QMessageBox.critical(self, self.tr("Preparation error"), str(error))

    def save(self):
        if self.preview_df is None:
            self.generate_preview()
        if self.preview_df is None:
            return
        if self.preview_df.attrs.get("data_status") != "ready":
            QMessageBox.warning(
                self, "Not ready", "Resolve missing values before saving."
            )
            return
        name = self.name.text().strip()
        if not name:
            QMessageBox.warning(self, self.tr("Invalid name"), self.tr("Enter an output name."))
            return
        self.prepared_data.emit(self.preview_df.copy(), name)
        QMessageBox.information(
            self,
            self.tr("Dataset prepared"),
            self.tr("'{name}' was added as READY.", name=name),
        )