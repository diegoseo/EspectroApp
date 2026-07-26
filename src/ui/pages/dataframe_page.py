import os
import numpy as np
import pandas as pd
from algorithms.preprocessing import get_column_with_fewest_rows
from functools import partial

from PySide6.QtCore import (
    Qt,
    QSize,
    Signal,
    QAbstractTableModel,
    QModelIndex,
)
from PySide6.QtGui import QIcon, QFont
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QFileDialog,
    QTableWidgetItem,
    QLabel,
    QDialog,
    QLineEdit,
    QGroupBox,
    QHeaderView,
    QScrollArea,
    QTableView,
    QSizePolicy,
    QFrame,
)


from core.translations import translate, get_language, retranslate_widget_tree


def tr(text, **values):
    return translate(text, get_language(), **values)


def normalize_visual_dataframe(df):
    """Return a readable copy of an internal-format spectral DataFrame."""
    df_out = df.copy()
    try:
        if df_out.empty:
            return df_out
        current_columns = [str(column) for column in df_out.columns]
        expected_columns = [str(index) for index in range(len(df_out.columns))]
        if current_columns == expected_columns:
            first_row = df_out.iloc[0].astype(str).tolist()
            df_out = df_out.iloc[1:].copy()
            df_out.columns = first_row
            df_out.reset_index(drop=True, inplace=True)
        return df_out
    except Exception:
        return df.copy()


class DataFrameInformationPage(QWidget):
    def __init__(
        self,
        df,
        file_name,
        back_callback,
        parent=None,
    ):
        super().__init__(parent)

        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setStyleSheet("""
            QWidget {
                background-color: #F8F7F3;
                color: #17231D;
                font-family: "Segoe UI", Arial, sans-serif;
            }

            QFrame#summaryCard,
            QFrame#typeCard {
                background-color: #FFFFFF;
                border: 1px solid #D8DDD9;
                border-radius: 11px;
            }

            QLabel#fileTitle {
                color: #17231D;
                font-size: 20px;
                font-weight: 700;
                background-color: transparent;
            }

            QLabel#summaryLabel {
                color: #607067;
                font-size: 14px;
                background-color: transparent;
            }

            QLabel#typeName {
                color: #17231D;
                font-size: 15px;
                font-weight: 600;
                background-color: transparent;
            }

            QLabel#typeCount {
                color: #0F8068;
                font-size: 15px;
                font-weight: 700;
                background-color: transparent;
            }

            QPushButton#backButton {
                background-color: #FFFFFF;
                color: #26332D;
                border: 1px solid #CFCFC8;
                border-radius: 7px;
                padding: 8px 16px;
                font-weight: 600;
            }

            QPushButton#backButton:hover {
                background-color: #F0EFEA;
            }
            """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        # Barra superior
        toolbar = QHBoxLayout()

        back_button = QPushButton(tr("← Back to loaded matrices"))
        back_button.setObjectName("backButton")
        back_button.clicked.connect(back_callback)

        toolbar.addWidget(back_button)
        toolbar.addStretch()

        main_layout.addLayout(toolbar)

        visible_name = os.path.basename(file_name)

        # Extraer nombres o tipos de muestra
        sample_labels = df.iloc[0, 1:].dropna().astype(str).str.strip()

        # Eliminar nombres vacíos
        sample_labels = sample_labels[sample_labels != ""]

        counts = sample_labels.value_counts(sort=False)

        total_samples = int(counts.sum())
        unique_types = int(len(counts))

        # Resumen general
        summary_card = QFrame()
        summary_card.setObjectName("summaryCard")

        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )
        summary_layout.setSpacing(7)

        file_title = QLabel(visible_name)
        file_title.setObjectName("fileTitle")

        summary_label = QLabel(
            tr(
                "Total samples: {total_samples}   ·   "
                "Sample types: {sample_types}   ·   "
                "Data points: {data_points}",
                total_samples=f"{total_samples:,}",
                sample_types=f"{unique_types:,}",
                data_points=f"{max(df.shape[0] - 1, 0):,}",
            )
        )
        summary_label.setObjectName("summaryLabel")

        summary_layout.addWidget(file_title)
        summary_layout.addWidget(summary_label)

        main_layout.addWidget(summary_card)

        # Lista desplazable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        types_layout = QVBoxLayout(container)
        types_layout.setContentsMargins(2, 2, 2, 2)
        types_layout.setSpacing(8)

        if counts.empty:
            empty_label = QLabel(tr("No sample labels were found in this dataset."))
            empty_label.setObjectName("summaryLabel")
            types_layout.addWidget(empty_label)

        else:
            for sample_type, quantity in counts.items():
                type_card = QFrame()
                type_card.setObjectName("typeCard")
                type_card.setMinimumHeight(58)

                card_layout = QHBoxLayout(type_card)
                card_layout.setContentsMargins(
                    18,
                    12,
                    18,
                    12,
                )

                type_name = QLabel(str(sample_type))
                type_name.setObjectName("typeName")

                type_count = QLabel(tr("{count} samples", count=f"{int(quantity):,}"))
                type_count.setObjectName("typeCount")

                card_layout.addWidget(type_name)
                card_layout.addStretch()
                card_layout.addWidget(type_count)

                types_layout.addWidget(type_card)

        types_layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll, 1)


class DataFrameSelectionWindow(QWidget):
    """
    Presents a scrollable list of loaded data matrices so the user can inspect or remove them.
    The window shows basic information for each DataFrame and forwards view/remove actions via callbacks to the main application.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        List of spectral DataFrames currently loaded in the session.
    file_names : list of str
        Corresponding file names or identifiers for each DataFrame in `dataframes`.
    eliminar_callback : callable
        Function that will be called with an index when the user chooses to remove a DataFrame.
    visualizar_callback : callable
        Function that will be called with an index when the user chooses to view a DataFrame in detail.
    """

    def __init__(
        self,
        dataframes,
        file_names,
        eliminar_callback,
        visualizar_callback,
        informacion_callback,
        back_callback=None,
        embedded=False,
    ):
        super().__init__()

        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.embedded = embedded
        self.dataframes = dataframes
        self.file_names = file_names
        self.eliminar_callback = eliminar_callback
        self.visualizar_callback = visualizar_callback
        self.informacion_callback = informacion_callback
        self.back_callback = back_callback

        self.setStyleSheet("""
            QWidget {
                background-color: #F8F7F3;
                color: #17231D;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 14px;
            }

            QLabel#windowTitle {
                background-color: transparent;
                color: #17231D;
                font-size: 25px;
                font-weight: 700;
                padding: 4px 0 2px 0;
            }
            QPushButton#backButton {
                background-color: #FFFFFF;
                color: #26332D;
                border: 1px solid #CFCFC8;
            }

            QPushButton#backButton:hover {
                background-color: #F0EFEA;
            }
            QLabel#windowSubtitle {
                background-color: transparent;
                color: #5F6F66;
                font-size: 14px;
                padding-bottom: 14px;
            }

            QScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollArea QWidget#qt_scrollarea_viewport {
                background-color: transparent;
            }

            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #D8DDD9;
                border-radius: 12px;
                margin: 0px;
            }

            QGroupBox:hover {
                border: 1px solid #AFC8BD;
            }

            QLabel#fileName {
                background-color: transparent;
                color: #17231D;
                font-size: 17px;
                font-weight: 700;
            }

            QLabel#fileInfo {
                background-color: transparent;
                color: #607067;
                font-size: 14px;
            }

            QPushButton {
                min-width: 105px;
                min-height: 34px;
                border-radius: 8px;
                padding: 7px 14px;
                font-size: 14px;
                font-weight: 600;
            }

            QPushButton#viewButton {
                background-color: #0F8068;
                color: #FFFFFF;
                border: 1px solid #0F8068;
            }

            QPushButton#viewButton:hover {
                background-color: #0B6E59;
                border: 1px solid #0B6E59;
            }

            QPushButton#infoButton {
                background-color: #FFFFFF;
                color: #25658A;
                border: 1px solid #9ABFD4;
            }

            QPushButton#infoButton:hover {
                background-color: #EEF7FC;
                color: #174F70;
                border: 1px solid #6FA5C3;
            }

            QPushButton#deleteButton {
                background-color: #FFFFFF;
                color: #A13F49;
                border: 1px solid #D8AEB3;
            }

            QPushButton#deleteButton:hover {
                background-color: #FFF1F2;
                color: #8B303A;
                border: 1px solid #C97C84;
            }
        """)

        main_layout = QVBoxLayout(self)

        if self.embedded:
            main_layout.setContentsMargins(4, 8, 4, 4)
        else:
            self.setWindowTitle(tr("Loaded data matrices"))
            self.setMinimumSize(850, 520)
            self.resize(900, 560)
            main_layout.setContentsMargins(28, 24, 28, 24)

        main_layout.setSpacing(10)

        if not self.embedded:
            title = QLabel(tr("Loaded data matrices"))
            title.setObjectName("windowTitle")
            title.setAlignment(Qt.AlignCenter)

            subtitle = QLabel(
                tr(
                    "Review the loaded datasets or remove those that are no longer needed."
                )
            )
            subtitle.setObjectName("windowSubtitle")
            subtitle.setAlignment(Qt.AlignCenter)
            subtitle.setWordWrap(True)

            main_layout.addWidget(title)
            main_layout.addWidget(subtitle)
            # Área desplazable para mostrar las matrices cargadas
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        contenedor_scroll = QWidget()
        layout_scroll = QVBoxLayout(contenedor_scroll)
        layout_scroll.setContentsMargins(4, 4, 4, 4)
        layout_scroll.setSpacing(12)

        for idx, (df, nombre) in enumerate(zip(self.dataframes, self.file_names)):
            grupo = QGroupBox()
            grupo.setMinimumHeight(115)
            grupo.setMaximumHeight(125)

            layout_grupo = QHBoxLayout(grupo)
            layout_grupo.setContentsMargins(20, 18, 20, 18)
            layout_grupo.setSpacing(18)

            nombre_visible = os.path.basename(nombre)
            n_filas, n_columnas = df.shape
            n_nulos = df.isnull().sum().sum()

            label_nombre = QLabel(nombre_visible)
            label_nombre.setObjectName("fileName")

            label_info = QLabel(
                tr(
                    "{rows} rows · {columns} columns · {nulls} null values",
                    rows=f"{n_filas:,}",
                    columns=f"{n_columnas:,}",
                    nulls=f"{n_nulos:,}",
                )
            )
            label_info.setObjectName("fileInfo")

            info_layout = QVBoxLayout()
            info_layout.setContentsMargins(0, 0, 0, 0)
            info_layout.setSpacing(6)
            info_layout.addWidget(label_nombre)
            info_layout.addWidget(label_info)

            boton_ver = QPushButton(tr("View"))
            boton_ver.setObjectName("viewButton")
            boton_ver.setFixedWidth(105)
            boton_ver.setIcon(QIcon("icom/view.png"))
            boton_ver.setIconSize(QSize(18, 18))
            boton_ver.setToolTip(tr("Open data matrix"))
            boton_ver.clicked.connect(partial(self.view_dataframe, idx))

            boton_info = QPushButton(tr("Information"))
            boton_info.setObjectName("infoButton")
            boton_info.setFixedWidth(130)
            boton_info.setToolTip(tr("View sample types and their quantities"))
            boton_info.clicked.connect(partial(self.show_information, idx))

            boton_borrar = QPushButton(tr("Remove"))
            boton_borrar.setObjectName("deleteButton")
            boton_borrar.setFixedWidth(110)
            boton_borrar.setIcon(QIcon("icom/delete.png"))
            boton_borrar.setIconSize(QSize(18, 18))
            boton_borrar.setToolTip(tr("Remove data matrix from the list"))
            boton_borrar.clicked.connect(partial(self.remove_dataframe, idx))

            botones_layout = QHBoxLayout()
            botones_layout.setContentsMargins(0, 0, 0, 0)
            botones_layout.setSpacing(8)

            botones_layout.addWidget(boton_ver)
            botones_layout.addWidget(boton_info)
            botones_layout.addWidget(boton_borrar)

            layout_grupo.addLayout(info_layout)
            layout_grupo.addStretch()
            layout_grupo.addLayout(botones_layout)

            layout_scroll.addWidget(grupo)

        layout_scroll.addStretch()

        scroll.setWidget(contenedor_scroll)
        main_layout.addWidget(scroll, 1)

        if self.embedded and self.back_callback is not None:
            back_layout = QHBoxLayout()
            back_layout.setContentsMargins(0, 8, 0, 0)

            back_button = QPushButton(tr("Back"))
            back_button.setObjectName("backButton")
            back_button.setFixedWidth(105)
            back_button.clicked.connect(self.back_callback)

            back_layout.addStretch()
            back_layout.addWidget(back_button)

            main_layout.addLayout(back_layout)

    def remove_dataframe(self, indice):
        """Show a fully translated, application-styled removal confirmation dialog."""
        dialog = QDialog(self)
        dialog.setModal(True)
        dialog.setWindowTitle(tr("Remove dataset"))
        dialog.setMinimumWidth(460)
        dialog.setObjectName("removeDatasetDialog")

        dialog.setStyleSheet("""
            QDialog#removeDatasetDialog {
                background-color: #F8F7F3;
                color: #17231D;
                font-family: "Segoe UI", Arial, sans-serif;
            }

            QLabel#removeDialogTitle {
                background-color: transparent;
                color: #17231D;
                font-size: 18px;
                font-weight: 700;
            }

            QLabel#removeDialogMessage {
                background-color: transparent;
                color: #52655B;
                font-size: 14px;
            }

            QPushButton {
                min-width: 122px;
                min-height: 40px;
                border-radius: 8px;
                padding: 7px 18px;
                font-size: 14px;
                font-weight: 600;
            }

            QPushButton#cancelRemoveButton {
                background-color: #FFFFFF;
                color: #26332D;
                border: 1px solid #CFCFC8;
            }

            QPushButton#cancelRemoveButton:hover {
                background-color: #F0EFEA;
                border-color: #AEB8B2;
            }

            QPushButton#cancelRemoveButton:pressed {
                background-color: #E7E5DF;
            }

            QPushButton#confirmRemoveButton {
                background-color: #C24B57;
                color: #FFFFFF;
                border: 1px solid #C24B57;
            }

            QPushButton#confirmRemoveButton:hover {
                background-color: #AB3F4A;
                border-color: #AB3F4A;
            }

            QPushButton#confirmRemoveButton:pressed {
                background-color: #91343E;
                border-color: #91343E;
            }
        """)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(26, 24, 26, 22)
        root.setSpacing(12)

        title = QLabel(tr("Remove dataset"))
        title.setObjectName("removeDialogTitle")

        message = QLabel(tr("Remove this dataset from the current session?"))
        message.setObjectName("removeDialogMessage")
        message.setWordWrap(True)

        root.addWidget(title)
        root.addWidget(message)
        root.addSpacing(8)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        buttons.addStretch()

        cancel_button = QPushButton(tr("Cancel"))
        cancel_button.setObjectName("cancelRemoveButton")
        cancel_button.setCursor(Qt.PointingHandCursor)
        cancel_button.clicked.connect(dialog.reject)

        remove_button = QPushButton(tr("Remove"))
        remove_button.setObjectName("confirmRemoveButton")
        remove_button.setCursor(Qt.PointingHandCursor)
        remove_button.clicked.connect(dialog.accept)
        remove_button.setDefault(True)

        buttons.addWidget(cancel_button)
        buttons.addWidget(remove_button)
        root.addLayout(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        self.eliminar_callback(indice)

        if not self.embedded:
            self.close()

    def view_dataframe(self, indice):
        self.visualizar_callback(indice)

        if not self.embedded:
            self.close()

    def show_information(self, indice):
        self.informacion_callback(indice)

        if not self.embedded:
            self.close()


class DataFrameFixWindow(QWidget):
    """
    Provides an interactive window to repair inconsistencies in a spectral DataFrame before further analysis.
    The user can trim rows, delete problematic columns, undo operations, preview the result, and export a cleaned matrix to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be inspected and corrected, typically containing spectral intensities and an X-axis column.
    """

    df_modificado = Signal(object)
    dataframe_exported = Signal(object, str)

    def __init__(self, df):
        super().__init__()
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("🛠 Fix DataFrame"))
        self.resize(600, 500)
        self.setStyleSheet("background-color: #2E2E2E; color: white;")

        self.df = df
        self.pila = [df.copy()]
        self.col, self.fil = get_column_with_fewest_rows(df)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        titulo = QLabel(tr("Modify DataFrame"))
        titulo.setFont(QFont("Arial", 15, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)

        grupo_botones = QGroupBox()
        grupo_botones.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444;
                border-radius: 10px;
                margin-top: 10px;
                background-color: #2b2b3d;
            }
        """)
        botones_layout = QVBoxLayout(grupo_botones)
        botones_layout.setSpacing(12)

        self.boton_fila = QPushButton(
            tr(
                "Remove rows from all DataFrames until they match the smallest one"
            )
        )
        self.boton_col = QPushButton(tr("Delete the column with the fewest rows"))
        self.boton_ver = QPushButton(tr("View current DataFrame"))
        self.boton_volver = QPushButton(tr("Restore previous state"))
        self.boton_csv = QPushButton(tr("Generate .CSV"))
        self.boton_salir = QPushButton(tr("Exit"))

        for b in [
            self.boton_fila,
            self.boton_col,
            self.boton_ver,
            self.boton_volver,
            self.boton_csv,
            self.boton_salir,
        ]:
            b.setStyleSheet("""
                QPushButton {
                    background-color: #004080;
                    color: white;
                    font-size: 14px;
                    padding: 10px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #0059b3;
                }
            """)
            botones_layout.addWidget(b)

        layout.addWidget(grupo_botones)

        self.boton_fila.clicked.connect(self.delete_rows)
        self.boton_col.clicked.connect(self.delete_column)
        self.boton_ver.clicked.connect(self.view_dataframe)
        self.boton_volver.clicked.connect(self.restore_previous_state)
        self.boton_csv.clicked.connect(self.generate_csv)
        self.boton_salir.clicked.connect(self.exit_window)

    def delete_rows(self):
        self.pila.append(self.df.copy())
        menor_cant_filas = self.df.dropna().shape[0]
        df_truncado = self.df.iloc[:menor_cant_filas]
        self.df = df_truncado

    def delete_column(self):
        self.pila.append(self.df.copy())
        col, _ = get_column_with_fewest_rows(self.df)
        self.df.drop(columns=[col], inplace=True)
        print(self.df)

    def view_dataframe(self):
        self.ventana_tabla = VerDf(self.df)
        self.ventana_tabla.show()

    def restore_previous_state(self):
        if len(self.pila) > 1:
            # Retrieve the previous state of the DataFrame
            self.df = self.pila.pop()
            print("The previous state has been restored.")
        else:
            print("There are no actions to undo.")

    def generate_csv(self):
        dialogo = FileNameDialog()

        if not dialogo.exec():
            print("Save canceled by the user.")
            return

        nombre = dialogo.get_name().strip()

        if not nombre:
            QMessageBox.warning(
                self,
                tr("Invalid name"),
                tr("Enter a name for the CSV file."),
            )
            return

        if not nombre.lower().endswith(".csv"):
            nombre += ".csv"

        try:
            df_exportar = normalize_visual_dataframe(self.df)

            df_exportar.to_csv(
                nombre,
                index=False,
            )

            self.dataframe_exported.emit(
                df_exportar.copy(),
                nombre,
            )

            print(f"File saved as: {nombre}")

        except Exception as error:
            QMessageBox.critical(
                self,
                tr("Export error"),
                tr(
                    "The DataFrame could not be exported:\n{error}",
                    error=error,
                ),
            )

    def exit_window(self):
        self.df_modificado.emit(self.df)
        self.close()


class PandasTableModel(QAbstractTableModel):
    """
    Modelo virtual para visualizar DataFrames grandes sin crear
    un QTableWidgetItem para cada celda.
    """

    def __init__(self, dataframe, parent=None):
        super().__init__(parent)
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self._df = dataframe

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return self._df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            value = self._df.iat[
                index.row(),
                index.column(),
            ]

            if pd.isna(value):
                return ""

            return str(value)

        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignCenter | Qt.AlignVCenter)

        return None

    def headerData(
        self,
        section,
        orientation,
        role=Qt.DisplayRole,
    ):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])

        return str(section + 1)


class VerDf(QWidget):
    def __init__(self, df):
        super().__init__()

        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("DataFrame View"))
        self.resize(1100, 750)

        layout = QVBoxLayout(self)

        df_mostrar = normalize_visual_dataframe(df)

        informacion = QLabel(
            tr(
                "{rows} rows × {columns} columns",
                rows=f"{df_mostrar.shape[0]:,}",
                columns=f"{df_mostrar.shape[1]:,}",
            )
        )
        layout.addWidget(informacion)

        self.tabla = QTableView()

        self.modelo = PandasTableModel(
            df_mostrar,
            self.tabla,
        )

        self.tabla.setModel(self.modelo)

        self.tabla.setAlternatingRowColors(True)
        self.tabla.setSortingEnabled(False)
        self.tabla.setWordWrap(False)

        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

        self.tabla.horizontalHeader().setDefaultSectionSize(115)

        self.tabla.verticalHeader().setDefaultSectionSize(24)

        layout.addWidget(self.tabla)


class FileNameDialog(QDialog):
    """
    Presents a simple dialog for entering the file name to use when saving a CSV export.
    The dialog collects a text string from the user, which calling code can retrieve and append an extension to as needed.
    """

    def __init__(self):
        super().__init__()
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("Save CSV"))
        self.setMinimumWidth(400)

        self.setStyleSheet("""
            QDialog {
                background-color: #2e2e2e;
                color: white;
                font-size: 14px;
                font-family: Segoe UI, Arial, sans-serif;
            }
            QLabel {
                margin-top: 10px;
                margin-bottom: 5px;
                color: white;
            }
            QLineEdit {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 6px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton#boton_cancelar {
                background-color: #f44336;
            }
            QPushButton#boton_cancelar:hover {
                background-color: #d32f2f;
            }
        """)

        layout = QVBoxLayout()
        self.label = QLabel(tr("File name:"))
        self.input = QLineEdit()
        layout.addWidget(self.label)
        layout.addWidget(self.input)

        botones = QHBoxLayout()
        self.cancel_button = QPushButton(tr("Cancel"))
        self.cancel_button.setObjectName("boton_cancelar")
        self.accept_button = QPushButton(tr("Accept"))
        self.accept_button.setObjectName("boton_aceptar")
        self.cancel_button.clicked.connect(self.reject)
        self.accept_button.clicked.connect(self.accept)
        botones.addWidget(self.accept_button)
        botones.addWidget(self.cancel_button)

        layout.addLayout(botones)
        self.setLayout(layout)

    def get_name(self):
        return self.input.text().strip()


class RamanRangeDialog(QDialog):
    """
    Collects a numeric Raman shift interval from the user to limit plots or CSV exports.
    The dialog validates that the minimum value is less than the maximum and exposes the accepted range through its attributes.

    """

    def __init__(self):
        super().__init__()
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("Raman shift range"))
        self.setMinimumWidth(350)

        layout = QVBoxLayout()

        self.setStyleSheet("""
            QDialog {
                background-color: #2e2e2e;
                color: white;
                font-size: 15px;
                font-family: Arial;
            }
            QLabel {
                margin-top: 8px;
                margin-bottom: 2px;
                color: white;
            }
            QLineEdit {
                background-color: #2e2e3e;
                color: white;
                border: 1px solid #5a5a7a;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                padding: 6px;
                border-radius: 4px;
                margin-top: 12px;
            }
            QPushButton:hover {
                background-color: #005f99;
            }
        """)

        self.label_min = QLabel(tr("Enter the minimum value:"))
        self.input_min = QLineEdit()
        layout.addWidget(self.label_min)
        layout.addWidget(self.input_min)

        self.label_max = QLabel(tr("Enter the maximum value:"))
        self.input_max = QLineEdit()
        layout.addWidget(self.label_max)
        layout.addWidget(self.input_max)

        self.accept_button = QPushButton(tr("Accept"))
        self.accept_button.clicked.connect(self.validate_and_submit)
        layout.addWidget(self.accept_button)

        self.setLayout(layout)

        self.valor_min = None
        self.valor_max = None

    def validate_and_submit(self):
        try:
            self.valor_min = float(self.input_min.text())
            self.valor_max = float(self.input_max.text())

            if self.valor_min >= self.valor_max:
                raise ValueError(
                    tr("The minimum value must be less than the maximum value.")
                )

            self.accept()
        except ValueError as e:
            QMessageBox.warning(
                self,
                tr("Error"),
                tr("Invalid input: {error}", error=e),
            )


class SampleTypeDialog(QDialog):
    """
    Collects a sample type or class name from the user to filter spectra for plotting or export.
    The dialog accepts free text input and exposes the chosen type through the `selected_type` attribute once accepted.
    """

    def __init__(self):
        super().__init__()
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("Plot Types"))
        self.setMinimumWidth(350)

        layout = QVBoxLayout()

        self.label_min = QLabel(tr("Enter the type you want to plot:"))
        self.label_min.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 5px;
            }
        """)

        self.input_min = QLineEdit()
        self.input_min.setPlaceholderText(tr("E.g.: ABSr"))
        self.input_min.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                border: 1px solid #2c3e50;
                border-radius: 4px;
                background-color: #1e272e;
                color: white;
            }
        """)

        self.accept_button = QPushButton(tr("Accept"))
        self.accept_button.setFixedHeight(36)
        self.accept_button.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #37a6f0;
            }
        """)
        self.accept_button.clicked.connect(self.validate_and_submit)

        layout.addWidget(self.label_min)
        layout.addWidget(self.input_min)
        layout.addWidget(self.accept_button)

        self.setStyleSheet("""
            QDialog {
                background-color: #2e2e2e;
            }
        """)

        self.setLayout(layout)

    def validate_and_submit(self):
        self.selected_type = self.input_min.text().strip()
        self.accept()


class LimitedRangeSampleTypeDialog(QDialog):
    """
    Collects both a sample type and a Raman shift interval from the user to filter spectra for plotting or export.
    The dialog validates the numeric range and exposes the chosen type and limits through its attributes after acceptance.
    """

    def __init__(self):
        super().__init__()
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("Plot Types"))
        self.setMinimumWidth(350)

        layout = QVBoxLayout()

        self.setStyleSheet("""
            QDialog {
                background-color: #2e2e2e;
                color: white;
                font-size: 15px;
                font-family: Segoe UI, Arial, sans-serif;
            }
            QLabel {
                margin-top: 8px;
                margin-bottom: 2px;
                color: white;
            }
            QLineEdit {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 5px;
                margin-top: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

        self.label_tipo = QLabel(tr("Enter the type you want to plot:"))
        self.input_tipo = QLineEdit()
        layout.addWidget(self.label_tipo)
        layout.addWidget(self.input_tipo)

        self.label_min = QLabel(tr("Enter the minimum value:"))
        self.input_min = QLineEdit()
        layout.addWidget(self.label_min)
        layout.addWidget(self.input_min)

        self.label_max = QLabel(tr("Enter the maximum value:"))
        self.input_max = QLineEdit()
        layout.addWidget(self.label_max)
        layout.addWidget(self.input_max)

        self.accept_button = QPushButton(tr("Accept"))
        self.accept_button.clicked.connect(self.validate_and_submit)
        layout.addWidget(self.accept_button)

        self.setLayout(layout)

        self.valor_min = None
        self.valor_max = None

    def validate_and_submit(self):
        try:
            self.selected_type = self.input_tipo.text().strip()
            self.valor_min = float(self.input_min.text())
            self.valor_max = float(self.input_max.text())

            if self.valor_min >= self.valor_max:
                raise ValueError(
                    tr("The minimum value must be less than the maximum value.")
                )

            self.accept()
        except ValueError as e:
            QMessageBox.warning(
                self,
                tr("Error"),
                tr("Invalid input: {error}", error=e),
            )


class CsvGenerator(QWidget):
    """
    Provides a minimal window for exporting a DataFrame to a CSV file with a user-chosen name.
    The widget uses a file name dialog and writes the DataFrame with headers, reporting success or failure via console messages.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be saved as a CSV file.
    """

    dataframe_exported = Signal(object, str)

    def __init__(self, df):
        super().__init__()
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("Fix DataFrame"))
        self.resize(300, 150)
        self.df = df
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

    def generar_csv(self):
        dialogo = FileNameDialog()

        if not dialogo.exec():
            print("Save canceled by the user.")
            return

        nombre = dialogo.get_name().strip()

        if not nombre:
            QMessageBox.warning(
                self,
                tr("Invalid name"),
                tr("Enter a name for the CSV file."),
            )
            return

        if not nombre.lower().endswith(".csv"):
            nombre += ".csv"

        try:
            self.df.to_csv(
                nombre,
                index=False,
                header=True,
            )

            self.dataframe_exported.emit(
                self.df.copy(),
                nombre,
            )

            print(f"File saved as: {nombre}")

        except Exception as error:
            QMessageBox.critical(
                self,
                tr("Export error"),
                tr(
                    "The DataFrame could not be exported:\n{error}",
                    error=error,
                ),
            )