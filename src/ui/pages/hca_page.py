import os
import re

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtCore import Qt
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QMessageBox,
    QLabel,
    QLineEdit,
    QGroupBox,
    QComboBox,
    QButtonGroup,
    QRadioButton,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QHeaderView,
    QScrollArea,
)

from thread import HcaThread


from core.translations import translate, get_language, retranslate_widget_tree


def tr(text, **values):
    return translate(text, get_language(), **values)


class VentanaHca(QWidget):
    """
    Displays a hierarchical cluster analysis (HCA) dendrogram inside a dedicated window.
    The widget embeds the provided Matplotlib figure in a FigureCanvas so users can visually inspect clustering relationships between spectra.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object containing the HCA dendrogram to display.
    parent : QWidget, optional
        Parent widget that will own this window, by default None.
    """

    def __init__(
        self,
        lista_df,
        file_names,
        menu_principal,
        embedded=False,
    ):
        super().__init__()

        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.embedded = embedded
        self.menu_principal = menu_principal
        if not self.embedded:
            self.setWindowTitle(tr("Hierarchical Cluster Analysis (HCA)"))
            self.setMinimumSize(520, 620)
            self.resize(520, 680)

        self.lista_df = lista_df.copy()
        self.nombres_archivos = file_names
        self.df = None

        self.setStyleSheet("""
            QWidget {
                background-color: #F8F7F3;
                color: #17231D;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
            }

            QLabel#windowTitle {
                color: #17231D;
                font-size: 23px;
                font-weight: 700;
                background-color: transparent;
            }

            QLabel#windowSubtitle {
                color: #5F6F66;
                font-size: 13px;
                background-color: transparent;
            }

            QLabel#fieldLabel {
                color: #17231D;
                font-weight: 600;
                background-color: transparent;
            }

            QLabel#categoryLabel {
                color: #879289;
                font-size: 10px;
                font-weight: 600;
                background-color: transparent;
                min-height: 16px;
            }

            QLabel#helpLabel {
                color: #879289;
                font-size: 10px;
                background-color: transparent;
                min-height: 18px;
            }

            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #DDDCD6;
                border-radius: 11px;
                margin-top: 14px;
                padding: 14px;
                font-weight: 700;
                color: #17231D;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 7px;
                color: #17231D;
                background-color: #F8F7F3;
            }

            QComboBox,
            QLineEdit {
                background-color: #FFFFFF;
                color: #17231D;
                border: 1px solid #D7D7D1;
                border-radius: 7px;
                padding: 7px 9px;
                min-height: 27px;
            }

            QComboBox:hover,
            QLineEdit:hover {
                border: 1px solid #7CAEF2;
            }

            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #17231D;
                selection-background-color: #CFE3FF;
                selection-color: #174A87;
            }

            QRadioButton {
                background-color: #FFFFFF;
                color: #17231D;
                border: 1px solid #DDDCD6;
                border-radius: 7px;
                padding: 8px 10px;
                spacing: 7px;
                min-height: 24px;
            }

            QRadioButton:hover {
                background-color: #F5F9FF;
                border: 1px solid #84B7FF;
            }

            QRadioButton:checked {
                background-color: #CFE3FF;
                color: #174A87;
                border: 1px solid #77AEF5;
            }

            QRadioButton:disabled {
                background-color: #F1F1EE;
                color: #A1A7A2;
                border: 1px solid #E0E0DA;
            }

            QPushButton {
                min-height: 34px;
                border-radius: 7px;
                padding: 7px 17px;
                font-weight: 600;
            }

            QPushButton#acceptButton {
                background-color: #0F8A6B;
                color: #FFFFFF;
                border: 1px solid #0F8A6B;
            }

            QPushButton#acceptButton:hover {
                background-color: #0D765D;
            }

            QPushButton#cancelButton {
                background-color: #FFFFFF;
                color: #27372F;
                border: 1px solid #D1D1CB;
            }

            QPushButton#cancelButton:hover {
                background-color: #F0EFEA;
            }
        """)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        content_widget = QWidget()
        content_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(18, 14, 18, 16)
        main_layout.setSpacing(14)

        scroll_area.setWidget(content_widget)
        outer_layout.addWidget(scroll_area)

        if not self.embedded:
            title = QLabel(tr("🌳 Hierarchical cluster analysis"))
            title.setObjectName("windowTitle")
            title.setAlignment(Qt.AlignCenter)

            subtitle = QLabel(
                tr(
                    "Select a spectral matrix, choose one distance metric "
                    "and one linkage method."
                )
            )
            subtitle.setObjectName("windowSubtitle")
            subtitle.setAlignment(Qt.AlignCenter)

            main_layout.addWidget(title)
            main_layout.addWidget(subtitle)

        dataset_group = QGroupBox(tr("Input dataset"))
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.setContentsMargins(22, 26, 22, 20)
        dataset_layout.setSpacing(6)

        dataset_label = QLabel(tr("Select a data matrix:"))
        dataset_label.setObjectName("fieldLabel")
        dataset_label.setWordWrap(True)
        dataset_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        self.selector_df = QComboBox()
        opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos]
        self.selector_df.addItems(opciones)
        self.selector_df.currentIndexChanged.connect(self.select_dataframe)

        dataset_layout.addWidget(dataset_label)
        dataset_layout.addWidget(self.selector_df)

        main_layout.addWidget(dataset_group)

        distance_group = QGroupBox(tr("▣ Distance metric"))
        distance_layout = QVBoxLayout(distance_group)
        distance_layout.setContentsMargins(22, 26, 22, 20)
        distance_layout.setSpacing(10)

        magnitude_label = QLabel(tr("BASED ON MAGNITUDE"))
        magnitude_label.setObjectName("categoryLabel")
        distance_layout.addWidget(magnitude_label)

        self.euclidiana = QRadioButton(tr("Euclidean"))
        self.manhattan = QRadioButton(tr("Manhattan"))
        self.chebyshev = QRadioButton(tr("Chebyshev"))

        magnitude_grid = QGridLayout()
        magnitude_grid.setHorizontalSpacing(12)
        magnitude_grid.setVerticalSpacing(8)
        magnitude_grid.addWidget(self.euclidiana, 0, 0)
        magnitude_grid.addWidget(self.manhattan, 0, 1)
        magnitude_grid.addWidget(self.chebyshev, 1, 0)
        magnitude_grid.setColumnStretch(0, 1)
        magnitude_grid.setColumnStretch(1, 1)

        distance_layout.addLayout(magnitude_grid)

        shape_label = QLabel(tr("BASED ON SHAPE / CORRELATION"))
        shape_label.setObjectName("categoryLabel")
        shape_label.setWordWrap(True)
        shape_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        distance_layout.addWidget(shape_label)

        self.coseno = QRadioButton(tr("Cosine"))
        self.correlación_pearson = QRadioButton(tr("Pearson"))
        self.correlación_spearman = QRadioButton(tr("Spearman"))

        shape_grid = QGridLayout()
        shape_grid.setHorizontalSpacing(12)
        shape_grid.setVerticalSpacing(8)
        shape_grid.addWidget(self.coseno, 0, 0)
        shape_grid.addWidget(self.correlación_pearson, 0, 1)
        shape_grid.addWidget(self.correlación_spearman, 1, 0)
        shape_grid.setColumnStretch(0, 1)
        shape_grid.setColumnStretch(1, 1)

        distance_layout.addLayout(shape_grid)

        self.metric_group = QButtonGroup(self)
        self.metric_group.setExclusive(True)

        metric_buttons = [
            self.euclidiana,
            self.manhattan,
            self.chebyshev,
            self.coseno,
            self.correlación_pearson,
            self.correlación_spearman,
        ]

        for index, button in enumerate(metric_buttons):
            self.metric_group.addButton(
                button,
                index,
            )

            button.toggled.connect(self.actualizar_estado_enlaces)

        main_layout.addWidget(distance_group)

        linkage_group = QGroupBox(tr("♧ Linkage method"))
        linkage_layout = QGridLayout(linkage_group)
        linkage_layout.setContentsMargins(22, 26, 22, 20)
        linkage_layout.setHorizontalSpacing(12)
        linkage_layout.setVerticalSpacing(8)

        self.ward = QRadioButton(tr("Ward"))
        self.single_linkage = QRadioButton(tr("Single"))
        self.complete_linkage = QRadioButton(tr("Complete"))
        self.average_linkage = QRadioButton(tr("Average"))

        self.linkage_group = QButtonGroup(self)
        self.linkage_group.setExclusive(True)

        linkage_buttons = [
            self.ward,
            self.single_linkage,
            self.complete_linkage,
            self.average_linkage,
        ]

        for index, button in enumerate(linkage_buttons):
            self.linkage_group.addButton(
                button,
                index,
            )
            row, column = divmod(index, 2)
            linkage_layout.addWidget(button, row, column)

        linkage_layout.setColumnStretch(0, 1)
        linkage_layout.setColumnStretch(1, 1)

        main_layout.addWidget(linkage_group)

        clustering_group = QGroupBox(tr("♧ Clustering options"))

        clustering_layout = QVBoxLayout(clustering_group)

        clustering_layout.setContentsMargins(
            16,
            20,
            16,
            14,
        )

        clustering_layout.setSpacing(5)

        clusters_label = QLabel(tr("Number of clusters (p) (default 12)"))
        clusters_label.setObjectName("fieldLabel")
        clusters_label.setWordWrap(True)
        clusters_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        self.input_clusters = QLineEdit()
        self.input_clusters.setText(tr("12"))
        self.input_clusters.setPlaceholderText(tr("12"))
        self.input_clusters.setMaximumWidth(225)

        clusters_help = QLabel(
            tr(
                "Used both to cut the tree (fcluster) and to truncate "
                "the dendrogram display."
            )
        )
        clusters_help.setObjectName("helpLabel")
        clusters_help.setWordWrap(True)
        clusters_help.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        clustering_layout.addWidget(clusters_label)
        clustering_layout.addWidget(self.input_clusters)
        clustering_layout.addWidget(clusters_help)

        main_layout.addWidget(clustering_group)

        self.euclidiana.setChecked(True)
        self.ward.setChecked(True)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        btn_aceptar = QPushButton(tr("Accept"))
        btn_aceptar.setObjectName("acceptButton")
        btn_aceptar.clicked.connect(self.apply_transformations_and_close)

        btn_cancelar = QPushButton(tr("Back"))
        btn_cancelar.setObjectName("cancelButton")

        if self.embedded:
            btn_cancelar.clicked.connect(self.return_to_main_page)
        else:
            btn_cancelar.clicked.connect(self.close)

        buttons_layout.addStretch()
        buttons_layout.addWidget(btn_cancelar)
        buttons_layout.addWidget(btn_aceptar)

        main_layout.addLayout(buttons_layout)

        if self.lista_df:
            self.select_dataframe(0)
        else:
            print("Empty list")

        self.actualizar_estado_enlaces()

    def return_to_main_page(self):
        if hasattr(self.menu_principal, "show_welcome_page"):
            self.menu_principal.show_welcome_page()
            return

        self.menu_principal.workspace_stack.setCurrentWidget(
            self.menu_principal.welcome_page
        )

    def select_dataframe(self, index):
        """
        Updates the currently selected DataFrame for HCA based on the index chosen in the combo box.
        The method stores a copy of the selected matrix and resolves its base file name for use in subsequent analysis or labeling.

        Parameters
        ----------
        index : int
            Zero-based index of the dataset in the internal list of loaded matrices.

        Returns
        -------
        None
        """
        self.df = self.lista_df[index].copy()
        nombre_archivo = os.path.basename(self.nombres_archivos[index])

    def return_to_hca_options(self):
        self.menu_principal.workspace_title.setText(tr("Hierarchical cluster analysis"))

        self.menu_principal.workspace_subtitle.setText(
            tr(
                "Select a spectral matrix, distance metric, linkage method "
                "and number of clusters."
            )
        )

        self.menu_principal.workspace_stack.setCurrentWidget(self)

    def apply_transformations_and_close(self):
        """
        Collects the HCA configuration from the UI and
        launches the clustering analysis in a background thread.

        Returns
        -------
        None
        """
        if self.df is None:
            QMessageBox.warning(
                self,
                tr("No selection"),
                tr("You must select a DataFrame."),
            )
            return

        try:
            numero_clusters = int(self.input_clusters.text().strip() or "12")

        except ValueError:
            QMessageBox.warning(
                self,
                tr("Invalid number of clusters"),
                tr("The number of clusters must be an integer."),
            )
            return

        if numero_clusters < 2:
            QMessageBox.warning(
                self,
                tr("Invalid number of clusters"),
                tr("The number of clusters must be at least 2."),
            )
            return

        numero_muestras = self.df.shape[1] - 1

        if numero_muestras < 2:
            QMessageBox.warning(
                self,
                tr("Insufficient samples"),
                tr("HCA requires at least two samples."),
            )
            return

        if numero_clusters > numero_muestras:
            QMessageBox.warning(
                self,
                tr("Invalid number of clusters"),
                tr(
                    "The selected dataset contains only {count} samples.",
                    count=numero_muestras,
                ),
            )
            return

        opciones = {}

        # Distance metric
        if self.euclidiana.isChecked():
            opciones["Euclidiana"] = True

        elif self.manhattan.isChecked():
            opciones["Manhattan"] = True

        elif self.coseno.isChecked():
            opciones["Coseno"] = True

        elif self.chebyshev.isChecked():
            opciones["Chebyshev"] = True

        elif self.correlación_pearson.isChecked():
            opciones["Correlación Pearson"] = True

        elif self.correlación_spearman.isChecked():
            opciones["Correlación Spearman"] = True

        else:
            QMessageBox.warning(
                self,
                tr("No distance metric"),
                tr("Select a distance metric."),
            )
            return

        # Linkage method
        if self.ward.isChecked():
            opciones["Ward"] = True

        elif self.single_linkage.isChecked():
            opciones["Single Linkage"] = True

        elif self.complete_linkage.isChecked():
            opciones["Complete Linkage"] = True

        elif self.average_linkage.isChecked():
            opciones["Average Linkage"] = True

        else:
            QMessageBox.warning(
                self,
                tr("No linkage method"),
                tr("Select a linkage method."),
            )
            return

        # Debe agregarse después de distancia y linkage.
        opciones["Numero Clusters"] = numero_clusters

        selected_index = self.selector_df.currentIndex()

        if selected_index >= 0 and selected_index < len(self.nombres_archivos):
            self._history_dataset_name = os.path.basename(
                str(self.nombres_archivos[selected_index])
            )
        else:
            self._history_dataset_name = "Unnamed dataset"

        self._history_options = {
            "distance": (
                "Euclidean"
                if self.euclidiana.isChecked()
                else (
                    "Manhattan"
                    if self.manhattan.isChecked()
                    else (
                        "Cosine"
                        if self.coseno.isChecked()
                        else (
                            "Chebyshev"
                            if self.chebyshev.isChecked()
                            else (
                                "Pearson"
                                if self.correlación_pearson.isChecked()
                                else "Spearman"
                            )
                        )
                    )
                )
            ),
            "linkage": (
                "Ward"
                if self.ward.isChecked()
                else (
                    "Single"
                    if self.single_linkage.isChecked()
                    else "Complete" if self.complete_linkage.isChecked() else "Average"
                )
            ),
            "clusters": numero_clusters,
        }

        self.hilo = HcaThread(
            self.df,
            opciones,
        )

        self.hilo.signal_resultado_hca.connect(self.generar_hca)

        self.hilo.error_signal.connect(self.show_hca_error)

        self.hilo.start()

    # IF EUCLIDEAN OR MANHATTAN IS NOT SELECTED, DISABLE WARD
    def actualizar_estado_enlaces(self):
        ward_allowed = self.euclidiana.isChecked() or self.manhattan.isChecked()

        self.ward.setEnabled(ward_allowed)

        if not ward_allowed and self.ward.isChecked():
            self.ward.setChecked(False)
            self.average_linkage.setChecked(True)

    def show_hca_error(self, message):
        self._history_options = None

        QMessageBox.critical(
            self,
            tr("HCA error"),
            tr(
                "The hierarchical cluster analysis could not be completed:\n{error}",
                error=message,
            ),
        )

    def generar_hca(self, fig, cluster_table):
        """Display the dendrogram and cluster composition in two tabs."""
        if hasattr(self.menu_principal, "record_analysis_step"):
            history_options = getattr(
                self,
                "_history_options",
                {},
            )

            self.menu_principal.record_analysis_step(
                dataset=getattr(
                    self,
                    "_history_dataset_name",
                    "Unnamed dataset",
                ),
                operation="Hierarchical cluster analysis",
                parameters={
                    "Distance metric": history_options.get(
                        "distance",
                        "Unknown",
                    ),
                    "Linkage method": history_options.get(
                        "linkage",
                        "Unknown",
                    ),
                    "Number of clusters": history_options.get(
                        "clusters",
                        "Unknown",
                    ),
                },
            )

        if hasattr(self.menu_principal, "hca_result_page"):
            old_page = self.menu_principal.hca_result_page
            old_index = self.menu_principal.workspace_stack.indexOf(old_page)
            if old_index != -1:
                self.menu_principal.workspace_stack.removeWidget(old_page)
            old_page.deleteLater()

        result_page = QWidget()
        result_layout = QVBoxLayout(result_page)
        result_layout.setContentsMargins(8, 8, 8, 8)
        result_layout.setSpacing(10)

        toolbar_layout = QHBoxLayout()
        back_button = QPushButton(tr("← Back to options"))
        back_button.setObjectName("backButton")
        back_button.clicked.connect(self.return_to_hca_options)

        result_title = QLabel(tr("HCA results"))
        result_title.setStyleSheet(
            "font-size: 19px; font-weight: 700; color: #17231D; "
            "background-color: transparent;"
        )
        toolbar_layout.addWidget(back_button)
        toolbar_layout.addWidget(result_title)
        toolbar_layout.addStretch()
        result_layout.addLayout(toolbar_layout)

        tabs = QTabWidget()

        dendrogram_tab = QWidget()
        dendrogram_layout = QVBoxLayout(dendrogram_tab)
        dendrogram_layout.setContentsMargins(6, 6, 6, 6)
        image_actions = QHBoxLayout()
        image_actions.addStretch()
        export_image_button = QPushButton(tr("Export image"))
        export_image_button.setObjectName("acceptButton")
        export_image_button.clicked.connect(lambda: self.export_hca_image(fig))
        image_actions.addWidget(export_image_button)
        dendrogram_layout.addLayout(image_actions)

        # Localize Matplotlib text generated by the clustering algorithm.
        try:
            for axis in fig.axes:
                title = axis.get_title()
                match = re.match(
                    r"Dendrogram using (.+?) linkage with (.+?) distance \(HCA\)", title
                )
                if match:
                    axis.set_title(
                        tr(
                            "Dendrogram using {linkage} linkage with {distance} distance (HCA)",
                            linkage=match.group(1),
                            distance=match.group(2),
                        )
                    )
                axis.set_xlabel(tr("Samples"))
                axis.set_ylabel(tr("Distance"))
        except Exception:
            pass

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.updateGeometry()
        dendrogram_layout.addWidget(canvas, 1)
        tabs.addTab(dendrogram_tab, tr("Dendrogram"))

        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_layout.setContentsMargins(6, 6, 6, 6)
        table_actions = QHBoxLayout()
        table_actions.addStretch()
        export_table_button = QPushButton(tr("Export table"))
        export_table_button.setObjectName("acceptButton")
        export_table_button.clicked.connect(
            lambda: self.export_cluster_table(cluster_table)
        )
        table_actions.addWidget(export_table_button)
        table_layout.addLayout(table_actions)

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(
            [tr("Cluster"), tr("Label"), tr("Size"), tr("Composition")]
        )
        table.setRowCount(len(cluster_table))
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)

        for row_index, row in cluster_table.iterrows():
            values = [row["Cluster"], row["Label"], row["Size"], row["Composition"]]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column_index < 3:
                    item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_index, column_index, item)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table_layout.addWidget(table, 1)
        tabs.addTab(table_tab, tr("Cluster composition"))
        result_layout.addWidget(tabs, 1)

        result_page.canvas = canvas
        result_page.cluster_table = cluster_table
        result_page.hca_figure = fig
        self.menu_principal.hca_result_page = result_page
        self.menu_principal.workspace_stack.addWidget(result_page)
        self.menu_principal.workspace_title.setText(
            tr("Hierarchical cluster analysis results")
        )
        self.menu_principal.workspace_subtitle.setText(
            tr("Inspect the dendrogram and the composition of each cluster.")
        )
        self.menu_principal.workspace_stack.setCurrentWidget(result_page)

    def export_hca_image(self, fig):
        """Export the dendrogram through Matplotlib."""
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            tr("Export HCA image"),
            "hca_dendrogram.png",
            tr(
                "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;"
                "SVG vector image (*.svg);;PDF document (*.pdf)"
            ),
        )
        if not path:
            return

        extension_map = {
            "PNG image (*.png)": ".png",
            "JPEG image (*.jpg *.jpeg)": ".jpg",
            "SVG vector image (*.svg)": ".svg",
            "PDF document (*.pdf)": ".pdf",
        }
        if not os.path.splitext(path)[1]:
            path += extension_map.get(selected_filter, ".png")

        try:
            fig.savefig(path, dpi=300, bbox_inches="tight")
            QMessageBox.information(
                self,
                tr("Image exported"),
                tr(
                    "The dendrogram was saved successfully:\n{path}",
                    path=path,
                ),
            )
        except Exception as error:
            QMessageBox.critical(
                self,
                tr("Export error"),
                tr(
                    "The dendrogram could not be exported:\n{error}",
                    error=error,
                ),
            )

    def export_cluster_table(self, cluster_table):
        """Export the cluster summary as CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("Export cluster composition"),
            "hca_cluster_composition.csv",
            tr("CSV file (*.csv)"),
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        try:
            cluster_table.to_csv(path, index=False, encoding="utf-8-sig")
            QMessageBox.information(
                self,
                tr("Table exported"),
                tr(
                    "The cluster table was saved successfully:\n{path}",
                    path=path,
                ),
            )
        except Exception as error:
            QMessageBox.critical(
                self,
                tr("Export error"),
                tr(
                    "The cluster table could not be exported:\n{error}",
                    error=error,
                ),
            )