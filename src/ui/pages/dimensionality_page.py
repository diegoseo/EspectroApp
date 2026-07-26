import os
import tempfile

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from functions import calculate_cumulative_variance, prepare_pca_matrix
from plotting import graficar_varianza_acumulada
from thread import DimensionalityReductionThread
from core.plotly_exporter import PlotlyExporter
from core.preprocessing_signature import dataset_pipeline_metadata

from core.translations import translate, get_language, retranslate_widget_tree


def tr(text, **values):
    return translate(text, get_language(), **values)


class MultivariateResultsPage(QWidget):
    def __init__(self, back_callback, parent=None):
        super().__init__(parent)

        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.temp_files = {}
        self.plot_views = {}
        self.figure_ids_by_title = {}

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(10)

        self.btn_back = QPushButton(tr("← Back to options"))
        self.btn_back.setObjectName("backButton")
        self.btn_back.setMinimumHeight(36)
        self.btn_back.clicked.connect(back_callback)

        toolbar_layout.addWidget(self.btn_back)
        toolbar_layout.addStretch()

        main_layout.addLayout(toolbar_layout)

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
                padding: 9px 18px;
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

    def activate_figure(self, fig):
        """Show the tab containing the requested figure."""
        figure_id = id(fig)

        for title, stored_figure_id in self.figure_ids_by_title.items():
            if stored_figure_id != figure_id:
                continue

            for index in range(self.tabs.count()):
                if self.tabs.tabText(index) == title:
                    self.tabs.setCurrentIndex(index)
                    return True

        return False

    def add_plot(self, fig, title):
        """
        Add or replace a Plotly or Matplotlib figure in a result tab.

        Plotly figures keep a reference to their QWebEngineView so they can be
        exported without Kaleido or an externally installed browser.
        """

        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == title:
                old_widget = self.tabs.widget(index)
                self.tabs.removeTab(index)
                old_widget.deleteLater()
                break

        old_figure_id = self.figure_ids_by_title.pop(title, None)
        if old_figure_id is not None:
            self.plot_views.pop(old_figure_id, None)

        old_temp_path = self.temp_files.pop(title, None)
        if old_temp_path:
            try:
                if os.path.exists(old_temp_path):
                    os.remove(old_temp_path)
            except OSError:
                pass

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Interactive Plotly figure.
        if hasattr(fig, "write_html"):
            browser = QWebEngineView(plot_container)
            plot_layout.addWidget(browser)

            temporary_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".html",
            )
            temporary_file.close()

            # Bundle plotly.js so visualization and export work offline.
            fig.write_html(
                temporary_file.name,
                include_plotlyjs=True,
                full_html=True,
                config={
                    "responsive": True,
                    "displaylogo": False,
                    "editable": True,
                    "edits": {
                        "annotationPosition": False,
                        "annotationTail": False,
                        "annotationText": False,
                        "axisTitleText": False,
                        "colorbarPosition": False,
                        "colorbarTitleText": False,
                        "legendPosition": True,
                        "legendText": False,
                        "shapePosition": False,
                        "titleText": False,
                    },
                },
            )

            browser.setUrl(QUrl.fromLocalFile(temporary_file.name))

            figure_id = id(fig)
            self.plot_views[figure_id] = browser
            self.figure_ids_by_title[title] = figure_id
            self.temp_files[title] = temporary_file.name

        # Static Matplotlib figure.
        elif isinstance(fig, Figure):
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Expanding,
            )
            canvas.updateGeometry()
            plot_layout.addWidget(canvas)

        else:
            raise TypeError(f"Unsupported figure type: {type(fig).__name__}")

        new_index = self.tabs.addTab(
            plot_container,
            title,
        )
        self.tabs.setCurrentIndex(new_index)

    def cleanup(self):
        for path in self.temp_files.values():
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

        self.temp_files.clear()
        self.plot_views.clear()
        self.figure_ids_by_title.clear()


class DimensionalityReductionWindow(QWidget):
    """
    Provides a configuration window for running PCA, t-SNE, and related dimensionality reduction analyses on spectral data.
    The window lets the user select a dataset, choose methods and components, configure plots and reports, and then executes the analysis in background threads while enabling figure export.

    Parameters
    ----------
    lista_df : list of pandas.DataFrame
        List of spectral DataFrames available for dimensionality reduction.
    file_names : list of str
        Names or paths associated with each DataFrame in `lista_df`, shown in the selection widget.
    menu_principal : QWidget
        Reference to the main menu window so analysis results and figures can be coordinated with the rest of the application.
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
        self.lista_df = lista_df.copy()
        self.file_names = file_names
        self.df = None

        if not self.embedded:
            self.setWindowTitle(tr("PCA and t-SNE Analysis"))
            self.setMinimumSize(900, 720)
            self.resize(1000, 760)

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
            }

            QLabel#windowSubtitle {
                background-color: transparent;
                color: #607067;
                font-size: 14px;
                padding-bottom: 8px;
            }

            QLabel#fieldLabel {
                background-color: transparent;
                color: #24372E;
                font-size: 13px;
                font-weight: 600;
                padding-bottom: 3px;
            }

            QLabel#smallFieldLabel {
                background-color: transparent;
                color: #76837C;
                font-size: 12px;
                font-weight: 500;
            }

            QScrollArea {
                background-color: transparent;
                border: none;
            }

            QScrollArea QWidget#qt_scrollarea_viewport {
                background-color: transparent;
            }

            QGroupBox#mainCard {
                background-color: #FFFFFF;
                border: 1px solid #DEDCD6;
                border-radius: 11px;
                margin-top: 18px;
                padding: 14px;
                color: #17231D;
                font-size: 14px;
                font-weight: 600;
            }

            QGroupBox#mainCard::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 7px;
                background-color: #F8F7F3;
                color: #173D31;
                font-size: 15px;
                font-weight: 700;
            }

            QFrame#methodCard {
                background-color: #FFFFFF;
                border: 1px solid #DEDCD6;
                border-radius: 8px;
            }

            QComboBox,
            QLineEdit {
                background-color: #FFFFFF;
                color: #17231D;
                border: 1px solid #D7D7D0;
                border-radius: 7px;
                padding: 7px 9px;
                min-height: 26px;
                font-size: 13px;
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

            QCheckBox {
                background-color: transparent;
                color: #27352E;
                padding: 4px 2px;
                font-size: 13px;
                font-weight: 500;
            }

            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #AEB8B2;
                border-radius: 3px;
                background-color: #FFFFFF;
                margin-right: 7px;
            }

            QCheckBox::indicator:checked {
                background-color: #E66D3C;
                border: 1px solid #E66D3C;
            }

            QPushButton {
                border-radius: 7px;
                padding: 7px 14px;
                min-height: 28px;
                font-size: 13px;
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
                border: 1px solid #AEB8B2;
            }

            QPushButton#analysisButton {
                background-color: #FFFFFF;
                color: #24372E;
                border: 1px solid #D0D0CA;
                text-align: left;
                padding-left: 12px;
            }

            QPushButton#analysisButton:hover {
                background-color: #DDF1EA;
                color: #155D4E;
                border: 1px solid #0F8068;
            }

            QPushButton#analysisButton:pressed {
                background-color: #CBE9DE;
            }

            QPushButton#analysisButton:disabled {
                background-color: #F2F2EE;
                color: #A0A8A3;
                border: 1px solid #E0E0DA;
            }
        """)

        main_layout = QVBoxLayout(self)

        if self.embedded:
            main_layout.setContentsMargins(4, 4, 4, 4)
        else:
            main_layout.setContentsMargins(22, 18, 22, 18)

        main_layout.setSpacing(10)

        if not self.embedded:
            title = QLabel(tr("PCA and t-SNE analysis"))
            title.setObjectName("windowTitle")
            title.setAlignment(Qt.AlignCenter)

            subtitle = QLabel(
                tr(
                    "Select a spectral matrix, choose dimensionality "
                    "reduction methods, and configure plots or reports."
                )
            )
            subtitle.setObjectName("windowSubtitle")
            subtitle.setAlignment(Qt.AlignCenter)
            subtitle.setWordWrap(True)

            main_layout.addWidget(title)
            main_layout.addWidget(subtitle)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(12)

        dataset_group = QGroupBox(tr("●  Input dataset"))
        dataset_group.setObjectName("mainCard")
        dataset_group.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.setContentsMargins(16, 16, 16, 14)
        dataset_layout.setSpacing(7)

        dataset_label = QLabel(tr("Select a data matrix for analysis:"))
        dataset_label.setObjectName("fieldLabel")

        self.selector_df = QComboBox()
        self.selector_df.setFixedHeight(38)

        opciones = [os.path.basename(nombre) for nombre in self.file_names]
        self.selector_df.addItems(opciones)
        self.selector_df.currentIndexChanged.connect(self.select_dataframe)

        dataset_layout.addWidget(dataset_label)
        dataset_layout.addWidget(self.selector_df)

        content_layout.addWidget(dataset_group)

        analysis_columns = QHBoxLayout()
        analysis_columns.setSpacing(12)

        left_column = QVBoxLayout()
        left_column.setSpacing(10)

        right_column = QVBoxLayout()
        right_column.setSpacing(10)

        methods_group = QGroupBox(tr("⌛  Dimensionality reduction"))
        methods_group.setObjectName("mainCard")

        methods_layout = QVBoxLayout(methods_group)
        methods_layout.setContentsMargins(14, 15, 14, 14)
        methods_layout.setSpacing(9)

        self.pca = QCheckBox(tr("PCA"))
        self.pca.setChecked(False)

        pca_card = QFrame()
        pca_card.setObjectName("methodCard")

        pca_layout = QVBoxLayout(pca_card)
        pca_layout.setContentsMargins(12, 10, 12, 12)
        pca_layout.setSpacing(6)

        self.label_reduccion_dim_componentes = QLabel(tr("Number of components"))
        self.label_reduccion_dim_componentes.setObjectName("smallFieldLabel")

        self.input_reduccion_dim_componentes = QLineEdit()
        self.input_reduccion_dim_componentes.setPlaceholderText(tr("E.g.: 2"))

        self.label_reduccion_dim_intervalo = QLabel(tr("Confidence interval (%)"))
        self.label_reduccion_dim_intervalo.setObjectName("smallFieldLabel")

        self.input_reduccion_dim_intervalo = QLineEdit()
        self.input_reduccion_dim_intervalo.setPlaceholderText(tr("E.g.: 90"))

        pca_layout.addWidget(self.label_reduccion_dim_componentes)
        pca_layout.addWidget(self.input_reduccion_dim_componentes)
        pca_layout.addWidget(self.label_reduccion_dim_intervalo)
        pca_layout.addWidget(self.input_reduccion_dim_intervalo)

        methods_layout.addWidget(self.pca)
        methods_layout.addWidget(pca_card)

        self.tsne = QCheckBox(tr("t-SNE"))
        self.tsne.setChecked(False)

        tsne_card = QFrame()
        tsne_card.setObjectName("methodCard")

        tsne_layout = QVBoxLayout(tsne_card)
        tsne_layout.setContentsMargins(12, 10, 12, 12)
        tsne_layout.setSpacing(6)

        tsne_dimensions_label = QLabel(tr("Output dimensions"))
        tsne_dimensions_label.setObjectName("smallFieldLabel")

        self.input_comp_tsne_direct = QLineEdit()
        self.input_comp_tsne_direct.setPlaceholderText(tr("E.g.: 2 or 3"))

        perplexity_label = QLabel(tr("Perplexity (default 30)"))
        perplexity_label.setObjectName("smallFieldLabel")

        self.input_perplexity = QLineEdit()
        self.input_perplexity.setPlaceholderText(tr("30"))

        iterations_label = QLabel(tr("Iterations (default 1000)"))
        iterations_label.setObjectName("smallFieldLabel")

        self.input_iterations_tsne = QLineEdit()
        self.input_iterations_tsne.setPlaceholderText(tr("1000"))

        methods_layout.addWidget(self.tsne)
        methods_layout.addWidget(tsne_card)

        tsne_bottom_row = QHBoxLayout()
        tsne_bottom_row.setSpacing(8)

        perplexity_layout = QVBoxLayout()
        perplexity_layout.setSpacing(4)
        perplexity_layout.addWidget(perplexity_label)
        perplexity_layout.addWidget(self.input_perplexity)

        iterations_layout = QVBoxLayout()
        iterations_layout.setSpacing(4)
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.input_iterations_tsne)

        tsne_bottom_row.addLayout(perplexity_layout)
        tsne_bottom_row.addLayout(iterations_layout)

        tsne_layout.addWidget(tsne_dimensions_label)
        tsne_layout.addWidget(self.input_comp_tsne_direct)
        tsne_layout.addLayout(tsne_bottom_row)

        self.tsne_pca = QCheckBox(tr("t-SNE(PCA(X))"))
        self.tsne_pca.stateChanged.connect(self.toggle_tsne_pca)

        self.input_comp_pca = QLineEdit()
        self.input_comp_pca.setPlaceholderText(
            tr("Number of PCs before t-SNE, e.g.: 10")
        )

        self.input_comp_tsne = QLineEdit()
        self.input_comp_tsne.setPlaceholderText(tr("Output dimensions, e.g.: 2 or 3"))

        self.input_perplexity_tsne_pca = QLineEdit()
        self.input_perplexity_tsne_pca.setPlaceholderText(tr("30"))

        self.input_iterations_tsne_pca = QLineEdit()
        self.input_iterations_tsne_pca.setPlaceholderText(tr("1000"))

        label_comp_pca = QLabel(tr("PCs before t-SNE"))
        label_comp_pca.setObjectName("smallFieldLabel")

        label_comp_tsne = QLabel(tr("t-SNE dimensions"))
        label_comp_tsne.setObjectName("smallFieldLabel")

        label_perplexity_pca = QLabel(tr("Perplexity (default 30)"))
        label_perplexity_pca.setObjectName("smallFieldLabel")

        label_iterations_pca = QLabel(tr("Iterations (default 1000)"))
        label_iterations_pca.setObjectName("smallFieldLabel")

        self.contenedor_componentes_tsne_pca = QWidget()
        layout_tsne_pca = QVBoxLayout(self.contenedor_componentes_tsne_pca)
        layout_tsne_pca.setContentsMargins(18, 2, 0, 2)
        layout_tsne_pca.setSpacing(6)

        top_tsne_pca_row = QHBoxLayout()
        top_tsne_pca_row.setSpacing(8)

        pca_before_layout = QVBoxLayout()
        pca_before_layout.setSpacing(4)
        pca_before_layout.addWidget(label_comp_pca)
        pca_before_layout.addWidget(self.input_comp_pca)

        dimensions_pca_layout = QVBoxLayout()
        dimensions_pca_layout.setSpacing(4)
        dimensions_pca_layout.addWidget(label_comp_tsne)
        dimensions_pca_layout.addWidget(self.input_comp_tsne)

        top_tsne_pca_row.addLayout(pca_before_layout)
        top_tsne_pca_row.addLayout(dimensions_pca_layout)

        bottom_tsne_pca_row = QHBoxLayout()
        bottom_tsne_pca_row.setSpacing(8)

        perplexity_pca_layout = QVBoxLayout()
        perplexity_pca_layout.setSpacing(4)
        perplexity_pca_layout.addWidget(label_perplexity_pca)
        perplexity_pca_layout.addWidget(self.input_perplexity_tsne_pca)

        iterations_pca_layout = QVBoxLayout()
        iterations_pca_layout.setSpacing(4)
        iterations_pca_layout.addWidget(label_iterations_pca)
        iterations_pca_layout.addWidget(self.input_iterations_tsne_pca)

        bottom_tsne_pca_row.addLayout(perplexity_pca_layout)
        bottom_tsne_pca_row.addLayout(iterations_pca_layout)

        layout_tsne_pca.addLayout(top_tsne_pca_row)
        layout_tsne_pca.addLayout(bottom_tsne_pca_row)

        self.contenedor_componentes_tsne_pca.hide()

        methods_layout.addWidget(self.tsne_pca)
        methods_layout.addWidget(self.contenedor_componentes_tsne_pca)

        left_column.addWidget(methods_group)

        plots_group = QGroupBox(tr("▥  Visualization outputs"))
        plots_group.setObjectName("mainCard")

        plots_layout = QVBoxLayout(plots_group)
        plots_layout.setContentsMargins(14, 15, 14, 14)
        plots_layout.setSpacing(5)

        self.grafico2d = QCheckBox(tr("2D score plot"))
        self.grafico3d = QCheckBox(tr("3D score plot"))
        self.graficoloading = QCheckBox(tr("PCA loading plot"))
        self.geninforme = QCheckBox(tr("Generate analysis report"))

        self.grafico2d.stateChanged.connect(self.toggle_gen2d)
        self.grafico3d.stateChanged.connect(self.toggle_gen3d)
        self.graficoloading.stateChanged.connect(self.toggle_loading)
        self.geninforme.stateChanged.connect(self.toggle_nombre_informe)

        self.input_x_2d = QLineEdit()
        self.input_x_2d.setPlaceholderText(tr("PC for X axis, e.g.: 1"))

        self.input_y_2d = QLineEdit()
        self.input_y_2d.setPlaceholderText(tr("PC for Y axis, e.g.: 2"))

        self.contenedor_componentes2d = QWidget()
        layout_2d = QVBoxLayout(self.contenedor_componentes2d)
        layout_2d.setContentsMargins(18, 2, 0, 2)
        layout_2d.setSpacing(5)
        layout_2d.addWidget(self.input_x_2d)
        layout_2d.addWidget(self.input_y_2d)

        self.contenedor_componentes2d.hide()

        self.input_x_3d = QLineEdit()
        self.input_x_3d.setPlaceholderText(tr("PC for X axis, e.g.: 1"))

        self.input_y_3d = QLineEdit()
        self.input_y_3d.setPlaceholderText(tr("PC for Y axis, e.g.: 2"))

        self.input_z_3d = QLineEdit()
        self.input_z_3d.setPlaceholderText(tr("PC for Z axis, e.g.: 3"))

        self.contenedor_componentes3d = QWidget()
        layout_3d = QVBoxLayout(self.contenedor_componentes3d)
        layout_3d.setContentsMargins(18, 2, 0, 2)
        layout_3d.setSpacing(5)
        layout_3d.addWidget(self.input_x_3d)
        layout_3d.addWidget(self.input_y_3d)
        layout_3d.addWidget(self.input_z_3d)

        self.contenedor_componentes3d.hide()

        self.contenedor_loading = QWidget()
        layout_loading = QVBoxLayout(self.contenedor_loading)
        layout_loading.setContentsMargins(18, 2, 0, 2)
        layout_loading.setSpacing(6)

        self.loading_help = QLabel(
            tr(
                "Select the principal components whose loading curves you want to display."
            )
        )
        self.loading_help.setObjectName("smallFieldLabel")
        self.loading_help.setWordWrap(True)
        layout_loading.addWidget(self.loading_help)

        self.loading_components_widget = QWidget()
        self.loading_components_layout = QGridLayout(self.loading_components_widget)
        self.loading_components_layout.setContentsMargins(0, 0, 0, 0)
        self.loading_components_layout.setHorizontalSpacing(12)
        self.loading_components_layout.setVerticalSpacing(4)

        self.loading_scroll = QScrollArea()
        self.loading_scroll.setWidgetResizable(True)
        self.loading_scroll.setWidget(self.loading_components_widget)
        self.loading_scroll.setMinimumHeight(100)
        self.loading_scroll.setMaximumHeight(180)
        layout_loading.addWidget(self.loading_scroll)

        self.loading_component_checkboxes = []
        self.loading_status = QLabel(tr("Choose the PCA component count first."))
        self.loading_status.setObjectName("smallFieldLabel")
        self.loading_status.setWordWrap(True)
        layout_loading.addWidget(self.loading_status)

        self.input_reduccion_dim_componentes.textChanged.connect(
            self.refresh_loading_component_options
        )

        self.contenedor_loading.hide()

        self.label_nombre_informe = QLabel(tr("Report file name"))
        self.label_nombre_informe.setObjectName("smallFieldLabel")

        self.input_nombre_informe = QLineEdit()
        self.input_nombre_informe.setPlaceholderText(tr("E.g.: report.txt"))

        self.contenedor_nombre_informe = QWidget()
        layout_report = QVBoxLayout(self.contenedor_nombre_informe)
        layout_report.setContentsMargins(18, 2, 0, 2)
        layout_report.setSpacing(5)
        layout_report.addWidget(self.label_nombre_informe)
        layout_report.addWidget(self.input_nombre_informe)

        self.contenedor_nombre_informe.hide()

        plots_layout.addWidget(self.grafico2d)
        plots_layout.addWidget(self.contenedor_componentes2d)

        plots_layout.addWidget(self.grafico3d)
        plots_layout.addWidget(self.contenedor_componentes3d)

        plots_layout.addWidget(self.graficoloading)
        plots_layout.addWidget(self.contenedor_loading)

        plots_layout.addWidget(self.geninforme)
        plots_layout.addWidget(self.contenedor_nombre_informe)

        right_column.addWidget(plots_group)

        export_group = QGroupBox(tr("⇩  Figure export"))
        export_group.setObjectName("mainCard")

        export_layout = QVBoxLayout(export_group)
        export_layout.setContentsMargins(14, 15, 14, 14)
        export_layout.setSpacing(6)

        self.btn_graficar_vacumulada = QPushButton(tr("View cumulative variance"))
        self.btn_save_pca2d = QPushButton(tr("Save PCA 2D"))
        self.btn_save_pca3d = QPushButton(tr("Save PCA 3D"))
        self.btn_save_tsne2d = QPushButton(tr("Save t-SNE 2D"))
        self.btn_save_tsne3d = QPushButton(tr("Save t-SNE 3D"))
        self.btn_save_loading = QPushButton(tr("Save loadings"))

        export_buttons = (
            self.btn_graficar_vacumulada,
            self.btn_save_pca2d,
            self.btn_save_pca3d,
            self.btn_save_tsne2d,
            self.btn_save_tsne3d,
            self.btn_save_loading,
        )

        for button in export_buttons:
            button.setObjectName("analysisButton")
            button.setMinimumHeight(31)
            export_layout.addWidget(button)

        for button in (
            self.btn_save_pca2d,
            self.btn_save_pca3d,
            self.btn_save_tsne2d,
            self.btn_save_tsne3d,
            self.btn_save_loading,
        ):
            button.setEnabled(False)

        right_column.addWidget(export_group)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content_widget)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_column.addStretch()
        right_column.addStretch()

        analysis_columns.addLayout(left_column, 1)
        analysis_columns.addLayout(right_column, 1)

        content_layout.addLayout(analysis_columns)
        content_layout.addStretch()

        main_layout.addWidget(scroll_area, 1)

        botones_layout = QHBoxLayout()
        botones_layout.setSpacing(10)

        btn_cancelar = QPushButton(tr("Back" if self.embedded else "Cancel"))
        btn_cancelar.setObjectName("cancelButton")
        btn_cancelar.setFixedWidth(100)

        btn_aceptar = QPushButton(tr("Accept"))
        btn_aceptar.setObjectName("acceptButton")
        btn_aceptar.setFixedWidth(110)

        btn_aceptar.clicked.connect(self.apply_transformations_and_close)

        if self.embedded:
            btn_cancelar.clicked.connect(self.menu_principal.show_welcome_page)
        else:
            btn_cancelar.clicked.connect(self.close)

        botones_layout.addStretch()
        botones_layout.addWidget(btn_cancelar)
        botones_layout.addWidget(btn_aceptar)

        main_layout.addLayout(botones_layout)

        self._fig_vacumulada = None
        self._fig_pca2d = None
        self._fig_pca3d = None
        self._fig_tsne2d = None
        self._fig_tsne3d = None
        self._fig_loading = None
        self._active_plotly_exporters = set()

        self.btn_graficar_vacumulada.clicked.connect(self._ver_varianza_acumulada)
        self.btn_save_pca2d.clicked.connect(
            lambda: self._guardar_fig(self._fig_pca2d, "pca_2d.png")
        )
        self.btn_save_pca3d.clicked.connect(
            lambda: self._guardar_fig(self._fig_pca3d, "pca_3d.png")
        )
        self.btn_save_tsne2d.clicked.connect(
            lambda: self._guardar_fig(self._fig_tsne2d, "tsne_2d.png")
        )
        self.btn_save_tsne3d.clicked.connect(
            lambda: self._guardar_fig(self._fig_tsne3d, "tsne_3d.png")
        )
        self.btn_save_loading.clicked.connect(
            lambda: self._guardar_fig(self._fig_loading, "loadings.png")
        )

        if self.selector_df.count() > 0:
            self.select_dataframe(0)

    def toggle_nombre_informe(self, state):
        """
        Shows or hides the report-name input container depending on whether report generation is enabled.
        The visibility directly follows the truthiness of the given state, so any truthy value makes the widgets visible and any falsy value hides them.

        Parameters
        ----------
        state : any
            Value indicating whether report generation is active; its boolean value controls the visibility of the name input container.

        Returns
        -------
        None
        """
        self.contenedor_nombre_informe.setVisible(bool(state))

    def toggle_gen2d(self, state):
        """
        Shows or hides the PCA/t-SNE 2D component selection fields based on the user's choice to generate a 2D plot.
        The visibility directly follows the truthiness of the given state, so any truthy value makes the inputs visible and any falsy value hides them.

        Parameters
        ----------
        state : any
            Value indicating whether a 2D score plot is requested; its boolean value controls the visibility of the 2D component widgets.

        Returns
        -------
        None
        """
        self.contenedor_componentes2d.setVisible(bool(state))

    def toggle_gen3d(self, state):
        """
        Shows or hides the PCA/t-SNE 3D component selection fields based on the user's choice to generate a 3D plot.
        The visibility directly follows the truthiness of the given state, so any truthy value makes the inputs visible and any falsy value hides them.

        Parameters
        ----------
        state : any
            Value indicating whether a 3D score plot is requested; its boolean value controls the visibility of the 3D component widgets.

        Returns
        -------
        None
        """
        self.contenedor_componentes3d.setVisible(bool(state))

    def toggle_tsne_pca(self, state):
        """
        Shows or hides the parameter fields for the t-SNE(PCA(X)) workflow based on the user's selection.
        The visibility directly follows the truthiness of the given state, so any truthy value makes the PCA and t-SNE component inputs visible and any falsy value hides them.

        Parameters
        ----------
        state : any
            Value indicating whether the t-SNE(PCA(X)) option is enabled; its boolean value controls the visibility of the parameter widgets.

        Returns
        -------
        None
        """
        self.contenedor_componentes_tsne_pca.setVisible(bool(state))

    def refresh_loading_component_options(self, *_):
        """Rebuild loading-PC checkboxes and show explained variance."""
        if not hasattr(self, "loading_components_layout"):
            return

        while self.loading_components_layout.count():
            item = self.loading_components_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.loading_component_checkboxes = []

        try:
            component_count = int(self.input_reduccion_dim_componentes.text().strip())
        except (TypeError, ValueError):
            component_count = 0

        if component_count < 2:
            self.loading_status.setText(
                "Enter at least 2 PCA components to configure the loading plot."
            )
            return

        variances = None
        if self.df is not None:
            try:
                result = calculate_cumulative_variance(self.df)
                variances = result[0]
            except Exception:
                variances = None

        available = component_count
        if variances is not None:
            available = min(available, len(variances))

        for index in range(available):
            pc_number = index + 1
            if variances is not None:
                label = f"PC{pc_number} — {float(variances[index]):.2f}%"
            else:
                label = f"PC{pc_number}"

            checkbox = QCheckBox(label)
            checkbox.setToolTip("Display this component as a loading curve.")
            if pc_number <= min(2, available):
                checkbox.setChecked(True)

            row = index // 2
            column = index % 2
            self.loading_components_layout.addWidget(checkbox, row, column)
            self.loading_component_checkboxes.append(checkbox)

        if available:
            self.loading_status.setText(
                "The percentage beside each PC is its explained variance. "
                "Interpret the loadings of PCs relevant to the score separation."
            )
        else:
            self.loading_status.setText(
                "No valid PCA components are available for this dataset."
            )

    def toggle_loading(self, state):
        """
        Shows or hides the PCA loading plot configuration fields based on the user's selection.
        The visibility directly follows the truthiness of the given state, so any truthy value makes the loading options visible and any falsy value hides them.

        Parameters
        ----------
        state : any
            Value indicating whether the loading plot option is enabled; its boolean value controls the visibility of the loading configuration widgets.

        Returns
        -------
        None
        """
        visible = bool(state)
        self.contenedor_loading.setVisible(visible)
        if visible:
            self.refresh_loading_component_options()

    def select_dataframe(self, index):
        """
        Updates the currently selected DataFrame for analysis based on a combo box index.
        The method safely ignores out-of-range indices and only updates the internal reference when a valid dataset position is chosen.

        Parameters
        ----------
        index : int
            Zero-based position of the chosen DataFrame in the internal list of loaded matrices.

        Returns
        -------
        None
        """
        if 0 <= index < len(self.lista_df):
            source_df = self.lista_df[index]
            self.df = source_df.copy()
            # Preserve dataset identity and preprocessing history explicitly.
            self.df.attrs = dict(getattr(source_df, "attrs", {}) or {})

    def prepare_results_page(self):
        """
        Creates a clean results page before starting
        a new multivariate analysis.
        """

        if hasattr(self, "results_page"):
            old_index = self.menu_principal.workspace_stack.indexOf(self.results_page)

            if old_index != -1:
                self.menu_principal.workspace_stack.removeWidget(self.results_page)

            self.results_page.cleanup()
            self.results_page.deleteLater()

        self.results_page = MultivariateResultsPage(
            back_callback=self.return_to_analysis_options
        )

        self.menu_principal.workspace_stack.addWidget(self.results_page)

    def return_to_analysis_options(self):
        self.menu_principal.workspace_title.setText(tr("PCA and t-SNE analysis"))

        self.menu_principal.workspace_subtitle.setText(
            "Select a spectral matrix and configure "
            "the multivariate analysis methods."
        )

        self.menu_principal.workspace_stack.setCurrentWidget(self)

    def _selected_dataset_name(self):
        """Return the visible name of the selected dataset."""
        index = self.selector_df.currentIndex()

        if index < 0 or index >= len(self.file_names):
            return "Unnamed dataset"

        return os.path.basename(str(self.file_names[index]))

    def _build_history_entry(
        self,
        opciones,
        componentes,
        intervalo,
        cp_pca,
        cp_tsne,
        tsne_parameters,
        componentes_selec,
        componentes_selec_loading,
    ):
        """Build a readable history entry for PCA and t-SNE."""
        operations = []
        parameters = {}

        if opciones.get("PCA"):
            operations.append("PCA analysis")
            parameters["PCA components"] = componentes
            parameters["Confidence interval"] = f"{intervalo:g}%"

        if opciones.get("TSNE"):
            operations.append("t-SNE analysis")
            parameters["t-SNE dimensions"] = tsne_parameters.get("direct_dimensions")
            parameters["t-SNE perplexity"] = tsne_parameters.get("direct_perplexity")
            parameters["t-SNE iterations"] = tsne_parameters.get("direct_iterations")

        if opciones.get("t-SNE(PCA(X))"):
            operations.append("PCA + t-SNE analysis")
            parameters["PCs before t-SNE"] = cp_pca
            parameters["t-SNE dimensions after PCA"] = cp_tsne
            parameters["PCA+t-SNE perplexity"] = tsne_parameters.get("pca_perplexity")
            parameters["PCA+t-SNE iterations"] = tsne_parameters.get("pca_iterations")

        if opciones.get("GRAFICO 2D"):
            parameters["2D axes"] = componentes_selec.get("2d")

        if opciones.get("GRAFICO 3D"):
            parameters["3D axes"] = componentes_selec.get("3d")

        if opciones.get("Grafico Loading (PCA)"):
            parameters["Loading components"] = componentes_selec_loading

        if opciones.get("GENERAR INFORME"):
            parameters["Report"] = "Generated"

        if not operations:
            operations.append("Multivariate analysis")

        return " + ".join(operations), parameters


    def _store_pending_pca_artifact(self, artifact):
        """Keep the fitted PCA object and a compact training snapshot."""
        artifact = dict(artifact or {})
        try:
            selected_index = self.selector_df.currentIndex()
            training_df = self.df

            # The PCA selector can contain a filtered/reordered dataset list.
            # Therefore its index must never be used directly against
            # menu_principal.dataframes, because that can select a different
            # (often raw) dataset and lose the preprocessing metadata.
            if 0 <= selected_index < len(self.lista_df):
                selected_df = self.lista_df[selected_index]
                selected_id = str(
                    getattr(selected_df, "attrs", {}).get("dataset_id", "")
                )
                training_df = selected_df

                # Prefer the live session object only when its stable ID matches.
                if selected_id:
                    for candidate in getattr(self.menu_principal, "dataframes", []):
                        candidate_id = str(
                            getattr(candidate, "attrs", {}).get("dataset_id", "")
                        )
                        if candidate_id == selected_id:
                            training_df = candidate
                            break

            training_matrix = prepare_pca_matrix(training_df)
            model = artifact.get("model")
            if model is not None:
                artifact["training_scores"] = model.transform(training_matrix)
            artifact["training_labels"] = list(
                training_df.attrs.get(
                    "class_labels",
                    training_df.iloc[0, 1:].astype(str).tolist(),
                )
            )
            artifact["training_sample_names"] = list(
                training_df.attrs.get(
                    "sample_ids",
                    [str(column) for column in training_df.columns[1:]],
                )
            )
            artifact["training_dataset_id"] = str(training_df.attrs.get("dataset_id", ""))
            artifact["training_dataset_name"] = self._selected_dataset_name()
            # Prefer the metadata snapshot captured before the worker started.
            # This avoids losing DataFrame.attrs across copied lists, queued Qt
            # signals, or legacy pages that reconstructed the selected frame.
            pipeline_metadata = dict(
                getattr(self, "_pending_training_pipeline_metadata", {}) or {}
            )
            if not pipeline_metadata:
                pipeline_metadata = dataset_pipeline_metadata(training_df)

            dataset_id = str(training_df.attrs.get("dataset_id", ""))
            registry = getattr(self.menu_principal, "dataset_metadata", {}) or {}
            registered_attrs = registry.get(dataset_id)
            if registered_attrs:
                registered_metadata = dataset_pipeline_metadata(
                    type("MetadataFrame", (), {"attrs": registered_attrs})()
                )
                if registered_metadata.get("options"):
                    pipeline_metadata = registered_metadata

            artifact["training_preprocessing"] = pipeline_metadata.get("options", {})
            artifact["training_preprocessing_signature"] = pipeline_metadata.get("signature", "")
            artifact["training_preprocessing_name"] = pipeline_metadata.get("name", "")
        except Exception as error:
            # Keep the fitted PCA reusable, but never silently lose the
            # preprocessing identity. The metadata captured before the worker
            # started is authoritative even if the training snapshot fails.
            pipeline_metadata = dict(
                getattr(self, "_pending_training_pipeline_metadata", {}) or {}
            )
            artifact["training_preprocessing"] = pipeline_metadata.get("options", {})
            artifact["training_preprocessing_signature"] = pipeline_metadata.get("signature", "")
            artifact["training_preprocessing_name"] = pipeline_metadata.get("name", "")
            artifact["training_snapshot_error"] = str(error)
        self._pending_pca_artifact = artifact

    def _record_completed_analysis(self):
        """Register the completed multivariate analysis in history."""
        history_data = getattr(
            self,
            "_pending_history_entry",
            None,
        )

        if not history_data:
            return

        if hasattr(self.menu_principal, "record_analysis_step"):
            self.menu_principal.record_analysis_step(
                dataset=history_data["dataset"],
                operation=history_data["operation"],
                parameters=history_data["parameters"],
            )

        # PCA creates a fitted transformation that can be registered and
        # restored as project metadata. t-SNE and HCA remain analysis runs,
        # not reusable predictive models.
        operation_text = str(history_data.get("operation", "")).lower()
        if "pca" in operation_text and hasattr(self.menu_principal, "record_fitted_model"):
            self.menu_principal.record_fitted_model(
                method_id="pca",
                dataset=history_data["dataset"],
                name=f"PCA — {history_data['dataset']}",
                parameters=history_data.get("parameters", {}),
                artifact=getattr(self, "_pending_pca_artifact", None),
            )
            self._pending_pca_artifact = None

        self._pending_history_entry = None

    def apply_transformations_and_close(self):
        """
        Gathers all dimensionality-reduction options from the UI and launches the analysis in a background thread.
        The method validates the current selection, builds the PCA/t-SNE configuration, connects result signals to plot windows, and then starts the worker thread.

        Returns
        -------
        None
        """
        componentes = self.input_reduccion_dim_componentes.text().strip() or "2"

        try:
            componentes = int(componentes)
        except ValueError:
            QMessageBox.warning(
                self,
                tr("Invalid PCA components"),
                tr("The number of PCA components must be an integer."),
            )
            return

        if componentes < 1:
            QMessageBox.warning(
                self,
                tr("Invalid PCA components"),
                tr("The number of PCA components must be greater than zero."),
            )
            return

        intervalo = self.input_reduccion_dim_intervalo.text().strip() or "90"

        try:
            intervalo_num = float(intervalo)
        except ValueError:
            QMessageBox.warning(
                self,
                tr("Invalid confidence interval"),
                tr("The confidence interval must be numeric."),
            )
            return

        if not 0 < intervalo_num < 100:
            QMessageBox.warning(
                self,
                tr("Invalid confidence interval"),
                tr(
                    "The confidence interval must be greater than 0 "
                    "and lower than 100."
                ),
            )
            return

        intervalo = intervalo_num

        nombre_informe = self.input_nombre_informe.text().strip()
        componentes_selec_loading = None
        cant_componentes_loading = 0

        if self.df is None:
            QMessageBox.warning(
                self,
                tr("No selection"),
                tr("You must select a DataFrame."),
            )
            return

        componentes_selec = {
            "2d": None,
            "3d": None,
        }
        opciones = {}

        cp_pca = None
        cp_tsne = None
        cp_tsne_direct = None
        tsne_perplexity = 30.0
        tsne_iterations = 1000
        tsne_pca_perplexity = 30.0
        tsne_pca_iterations = 1000

        if self.pca.isChecked():
            if self.grafico3d.isChecked() and componentes < 3:
                QMessageBox.warning(
                    self,
                    tr("Insufficient PCA components"),
                    tr("A 3D PCA plot requires at least 3 principal components."),
                )
                return
            opciones["PCA"] = True
        try:
            if self.tsne.isChecked():
                opciones["TSNE"] = True

                cp_tsne_direct = int(self.input_comp_tsne_direct.text() or 2)
                tsne_perplexity = float(self.input_perplexity.text() or 30)
                tsne_iterations = int(self.input_iterations_tsne.text() or 1000)
            if self.tsne_pca.isChecked():
                opciones["t-SNE(PCA(X))"] = True

                cp_pca = int(self.input_comp_pca.text() or 10)
                cp_tsne = int(self.input_comp_tsne.text() or 2)

                tsne_pca_perplexity = float(self.input_perplexity_tsne_pca.text() or 30)
                tsne_pca_iterations = int(self.input_iterations_tsne_pca.text() or 1000)
        except ValueError:
            QMessageBox.warning(
                self,
                tr("Invalid parameters"),
                tr(
                    "Dimensions and iterations must be integers, "
                    "and perplexity must be numeric."
                ),
            )
            return

        try:
            if self.grafico2d.isChecked():
                opciones["GRAFICO 2D"] = True

                pc_x = int(self.input_x_2d.text() or 1)
                pc_y = int(self.input_y_2d.text() or 2)

                componentes_selec["2d"] = [pc_x, pc_y]

            if self.grafico3d.isChecked():
                opciones["GRAFICO 3D"] = True

                pc_x = int(self.input_x_3d.text() or 1)
                pc_y = int(self.input_y_3d.text() or 2)
                pc_z = int(self.input_z_3d.text() or 3)

                componentes_selec["3d"] = [
                    pc_x,
                    pc_y,
                    pc_z,
                ]

        except ValueError:
            QMessageBox.warning(
                self,
                tr("Invalid plot components"),
                tr("The plot component numbers must be integers."),
            )
            return

        if self.geninforme.isChecked():
            opciones["GENERAR INFORME"] = True

        if self.graficoloading.isChecked():
            opciones["Grafico Loading (PCA)"] = True

            componentes_selec_loading = [
                index + 1
                for index, checkbox in enumerate(self.loading_component_checkboxes)
                if checkbox.isChecked()
            ]

            if not componentes_selec_loading:
                QMessageBox.warning(
                    self,
                    tr("No loading components selected"),
                    tr(
                        "Select at least one principal component "
                        "for the loading plot."
                    ),
                )
                return

            if any(component > componentes for component in componentes_selec_loading):
                QMessageBox.warning(
                    self,
                    tr("Invalid loading components"),
                    tr(
                        "Loading components must be between 1 and {maximum}.",
                        maximum=componentes,
                    ),
                )
                return

            cant_componentes_loading = max(componentes_selec_loading)

        tsne_parameters = {
            "direct_dimensions": cp_tsne_direct,
            "direct_perplexity": tsne_perplexity,
            "direct_iterations": tsne_iterations,
            "pca_perplexity": tsne_pca_perplexity,
            "pca_iterations": tsne_pca_iterations,
        }

        if not any(
            (
                self.pca.isChecked(),
                self.tsne.isChecked(),
                self.tsne_pca.isChecked(),
            )
        ):
            QMessageBox.warning(
                self,
                tr("No method selected"),
                tr("Select PCA, t-SNE or t-SNE(PCA(X))."),
            )
            return

        if self.tsne.isChecked() and cp_tsne_direct not in (2, 3):
            QMessageBox.warning(
                self,
                tr("Invalid dimensions"),
                tr("Direct t-SNE dimensions must be 2 or 3."),
            )
            return

        if self.tsne_pca.isChecked() and cp_tsne not in (2, 3):
            QMessageBox.warning(
                self,
                tr("Invalid dimensions"),
                tr("t-SNE(PCA(X)) dimensions must be 2 or 3."),
            )
            return

        if self.pca.isChecked():
            if self.grafico2d.isChecked():
                if componentes < 2:
                    QMessageBox.warning(
                        self,
                        tr("Insufficient PCA components"),
                        tr("A 2D PCA plot requires at least 2 principal components."),
                    )
                    return
                selected_2d = componentes_selec["2d"]

                if any(
                    component < 1 or component > componentes
                    for component in selected_2d
                ):
                    QMessageBox.warning(
                        self,
                        tr("Invalid PCA components"),
                        tr(
                            "The 2D plot components must be between 1 and {maximum}.",
                            maximum=componentes,
                        ),
                    )
                    return

                if len(set(selected_2d)) != 2:
                    QMessageBox.warning(
                        self,
                        tr("Repeated PCA components"),
                        tr("The X and Y components must be different."),
                    )
                    return

            if self.grafico3d.isChecked():
                selected_3d = componentes_selec["3d"]

                if any(
                    component < 1 or component > componentes
                    for component in selected_3d
                ):
                    QMessageBox.warning(
                        self,
                        tr("Invalid PCA components"),
                        tr(
                            "The 3D plot components must be between 1 and {maximum}.",
                            maximum=componentes,
                        ),
                    )
                    return

                if len(set(selected_3d)) != 3:
                    QMessageBox.warning(
                        self,
                        tr("Repeated PCA components"),
                        tr("The X, Y and Z components must be different."),
                    )
                    return

        history_operation, history_parameters = self._build_history_entry(
            opciones,
            componentes,
            intervalo,
            cp_pca,
            cp_tsne,
            tsne_parameters,
            componentes_selec,
            componentes_selec_loading,
        )

        self._pending_history_entry = {
            "dataset": self._selected_dataset_name(),
            "operation": history_operation,
            "parameters": history_parameters,
        }

        # Capture preprocessing metadata synchronously, before the worker thread
        # starts. This is the authoritative source for the fitted PCA record.
        selected_index = self.selector_df.currentIndex()
        selected_df = self.df
        if 0 <= selected_index < len(self.lista_df):
            selected_df = self.lista_df[selected_index]
        self._pending_training_pipeline_metadata = dataset_pipeline_metadata(selected_df)

        self.hilo = DimensionalityReductionThread(
            self.df,
            opciones,
            componentes,
            intervalo,
            nombre_informe,
            componentes_selec,
            cp_pca,
            cp_tsne,
            componentes_selec_loading,
            cant_componentes_loading,
            tsne_parameters,
        )

        self.hilo.pca_2d_figure_signal.connect(self.mostrar_grafico_pca_2d)
        self.hilo.pca_model_signal.connect(self._store_pending_pca_artifact)
        self.hilo.pca_3d_figure_signal.connect(self.mostrar_grafico_pca_3d)
        self.hilo.tsne_2d_figure_signal.connect(self.mostrar_grafico_tsne_2d)
        self.hilo.tsne_3d_figure_signal.connect(self.mostrar_grafico_tsne_3d)
        self.hilo.loading_figure_signal.connect(self.mostrar_grafico_loading)
        self.hilo.tsne_pca_2d_figure_signal.connect(self.mostrar_grafico_tsne_pca_2d)
        self.hilo.tsne_pca_3d_figure_signal.connect(self.mostrar_grafico_tsne_pca_3d)

        self.hilo.error_signal.connect(self.show_analysis_error)
        self.hilo.finished.connect(self._record_completed_analysis)

        self.prepare_results_page()

        self.menu_principal.workspace_title.setText(tr("Multivariate analysis results"))

        self.menu_principal.workspace_subtitle.setText(
            tr("Interactive PCA, t-SNE and loading plots.")
        )

        self.menu_principal.workspace_stack.setCurrentWidget(self.results_page)

        self.hilo.start()

    def show_analysis_error(self, message):
        self._pending_history_entry = None

        QMessageBox.critical(
            self,
            tr("Analysis error"),
            message,
        )

    def _guardar_fig(self, fig, nombre_defecto):
        """
        Export Plotly or Matplotlib figures without Kaleido.

        Plotly:
        - PNG/SVG through Plotly.js running in Qt WebEngine
        - PDF through QWebEnginePage.printToPdf()
        - HTML through Plotly's self-contained HTML writer

        Matplotlib:
        - PNG/SVG/PDF through Figure.savefig()
        """
        if fig is None:
            QMessageBox.warning(
                self,
                tr("Warning"),
                tr("There is no figure to save."),
            )
            return

        ruta, selected_filter = QFileDialog.getSaveFileName(
            self,
            tr("Save plot"),
            nombre_defecto,
            tr("PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;HTML (*.html)"),
        )
        if not ruta:
            return

        filter_extensions = {
            "PNG": ".png",
            "SVG": ".svg",
            "PDF": ".pdf",
            "HTML": ".html",
        }

        selected_extension = next(
            (
                extension
                for filter_name, extension in filter_extensions.items()
                if selected_filter.startswith(filter_name)
            ),
            None,
        )

        path_root, typed_extension = os.path.splitext(ruta)
        typed_extension = typed_extension.lower()

        # The selected filter is authoritative. This prevents a default name
        # such as "pca_2d.png" from forcing PNG when SVG, PDF or HTML is chosen.
        extension = selected_extension or typed_extension or ".png"

        if typed_extension != extension:
            ruta = path_root + extension
        elif not typed_extension:
            ruta += extension

        if isinstance(fig, Figure):
            if extension not in {".png", ".svg", ".pdf"}:
                QMessageBox.warning(
                    self,
                    tr("Unsupported figure type"),
                    tr("Matplotlib figures can be exported as PNG, SVG or PDF."),
                )
                return

            try:
                fig.savefig(
                    ruta,
                    dpi=300,
                    bbox_inches="tight",
                )
                QMessageBox.information(
                    self,
                    tr("Success"),
                    tr("Plot saved to:\n{path}", path=ruta),
                )
            except Exception as error:
                QMessageBox.critical(
                    self,
                    tr("Export error"),
                    str(error),
                )
            return

        if extension == ".html":
            try:
                fig.write_html(
                    ruta,
                    include_plotlyjs=True,
                    full_html=True,
                    config={
                        "responsive": True,
                        "displaylogo": False,
                    },
                )
                QMessageBox.information(
                    self,
                    tr("Success"),
                    tr("Plot saved to:\n{path}", path=ruta),
                )
            except Exception as error:
                QMessageBox.critical(
                    self,
                    tr("Export error"),
                    str(error),
                )
            return

        if extension not in {".png", ".svg", ".pdf"}:
            QMessageBox.warning(
                self,
                tr("Unsupported figure type"),
                tr("Select PNG, SVG, PDF or HTML."),
            )
            return

        if not hasattr(self, "results_page"):
            QMessageBox.critical(
                self,
                tr("Export error"),
                tr("The results page is not available."),
            )
            return

        self.menu_principal.workspace_stack.setCurrentWidget(self.results_page)

        self.results_page.activate_figure(fig)

        web_view = self.results_page.plot_views.get(id(fig))

        if web_view is None:
            QMessageBox.critical(
                self,
                tr("Export error"),
                tr("The rendered Plotly view was not found."),
            )
            return

        QTimer.singleShot(
            800,
            lambda: self._start_plotly_export(
                web_view,
                ruta,
                extension,
            ),
        )

    def _start_plotly_export(
        self,
        web_view,
        ruta,
        extension,
    ):
        """Start export after the Plotly view has become visible."""

        if web_view is None or web_view.page() is None:
            QMessageBox.critical(
                self,
                tr("Export error"),
                tr("The Plotly web view is not available."),
            )
            return

        exporter = PlotlyExporter(
            web_view,
            self,
        )

        self._active_plotly_exporters.add(exporter)

        exporter.export_finished.connect(
            lambda path, item=exporter: self._plot_export_finished(item, path)
        )

        exporter.export_error.connect(
            lambda message, item=exporter: self._plot_export_error(item, message)
        )

        if extension == ".pdf":
            exporter.export_pdf(ruta)

        else:
            exporter.export_image(
                ruta,
                image_format=extension.lstrip("."),
                width=1600,
                height=1000,
                scale=2,
            )

    def _plot_export_finished(self, exporter, file_path):
        self._active_plotly_exporters.discard(exporter)
        exporter.deleteLater()
        QMessageBox.information(
            self,
            tr("Success"),
            tr("Plot saved to:\n{path}", path=file_path),
        )

    def _plot_export_error(self, exporter, message):
        self._active_plotly_exporters.discard(exporter)
        exporter.deleteLater()
        QMessageBox.critical(
            self,
            tr("Export error"),
            str(message),
        )

    def _ver_varianza_acumulada(self):
        """
        Computes and displays the cumulative explained variance curve for the currently selected PCA dataset.
        The method estimates how many components are needed to reach a 95% threshold, shows the curve in a figure, stores it for export, and reports the required number of PCs to the user.

        Returns
        -------
        None
        """
        if self.df is None:
            QMessageBox.warning(
                self,
                tr("No data"),
                tr("No data has been loaded."),
            )
            return

        try:
            var_ind, var_acum, n95 = calculate_cumulative_variance(self.df, umbral=95)
            max_cp = n95 * 3
            fig = graficar_varianza_acumulada(
                var_acum, var_ind=var_ind, umbral=95, max_cp=max_cp, anotar=True
            )

            self._fig_vacumulada = fig
            fig.show()

            QMessageBox.information(
                self,
                tr("PCA"),
                tr("PCs required for ≥95%: {count}", count=n95),
            )

        except Exception as e:
            QMessageBox.critical(self, tr("Error"), str(e))

    def mostrar_grafico_pca_2d(self, fig):
        print("[GUI] Señal PCA 2D recibida.")

        self._fig_pca2d = fig
        self.btn_save_pca2d.setEnabled(True)

        self.results_page.add_plot(
            fig,
            "PCA 2D",
        )

        print("[GUI] PCA 2D agregado a resultados.")

    def mostrar_grafico_pca_3d(self, fig):
        self._fig_pca3d = fig
        self.btn_save_pca3d.setEnabled(True)

        self.results_page.add_plot(
            fig,
            "PCA 3D",
        )

    def mostrar_grafico_tsne_2d(self, fig):
        self._fig_tsne2d = fig
        self.btn_save_tsne2d.setEnabled(True)

        self.results_page.add_plot(
            fig,
            "t-SNE 2D",
        )

    def mostrar_grafico_tsne_3d(self, fig):
        self._fig_tsne3d = fig
        self.btn_save_tsne3d.setEnabled(True)

        self.results_page.add_plot(
            fig,
            "t-SNE 3D",
        )

    def mostrar_grafico_tsne_pca_2d(self, fig):
        self._fig_tsne_pca_2d = fig

        self.results_page.add_plot(
            fig,
            "PCA+t-SNE 2D",
        )

    def mostrar_grafico_tsne_pca_3d(self, fig):
        self._fig_tsne_pca_3d = fig

        self.results_page.add_plot(
            fig,
            "PCA+t-SNE 3D",
        )

    def mostrar_grafico_loading(self, fig):
        self._fig_loading = fig
        self.btn_save_loading.setEnabled(True)

        self.results_page.add_plot(
            fig,
            "PCA loadings",
        )


class VentanaGraficoPCA2D(QWidget):
    """
    Displays a 2D PCA Plotly figure inside an embedded web view window.
    The widget writes the figure to a temporary HTML file, loads it in a QWebEngineView, and cleans up the file when the window is closed.
    """

    def __init__(self, fig, parent=None):
        super().__init__(parent)
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("2D PCA Plot"))

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.setUrl(QUrl.fromLocalFile(f.name))

        self.tempfile_path = f.name

    def closeEvent(self, event):
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()


class VentanaGraficoPCA3D(QWidget):
    """
    Displays a 3D PCA Plotly figure inside an embedded web view window.
    The widget writes the figure to a temporary HTML file, loads it in a QWebEngineView for interactive viewing, and removes the file when the window is closed.
    """

    def __init__(self, fig, parent=None):
        super().__init__(parent)
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("3D PCA Plot"))

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.setUrl(QUrl.fromLocalFile(f.name))
            self.tempfile_path = f.name

    def closeEvent(self, event):
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()


class VentanaGraficoTSNE2D(QWidget):
    """
    Displays a 2D t-SNE Plotly figure inside an embedded web view window.
    The widget writes the figure to a temporary HTML file, loads it in a QWebEngineView, and deletes the file when the window is closed.
    """

    def __init__(self, fig, parent=None):
        super().__init__(parent)
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("2D t-SNE Plot"))

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.setUrl(QUrl.fromLocalFile(f.name))
            self.tempfile_path = f.name

    def closeEvent(self, event):
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()


class VentanaGraficoTSNE3D(QWidget):
    """
    Displays a 3D t-SNE Plotly figure inside an embedded web view window.
    The widget writes the figure to a temporary HTML file, loads it in a QWebEngineView for interactive exploration, and deletes the file when the window is closed.

    Parameters
    ----------
    fig : object
        Plotly figure to be rendered inside the window.
    parent : QWidget, optional
        Parent widget that will own this window, by default None.
    """

    def __init__(self, fig, parent=None):
        super().__init__(parent)
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("3D t-SNE Plot"))
        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.setUrl(QUrl.fromLocalFile(f.name))
            self.tempfile_path = f.name

    def closeEvent(self, event):
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()


class VentanaGraficoLoading(QWidget):
    """
    Displays a static PCA loading plot inside a dedicated window.
    The widget embeds the provided Matplotlib figure in a FigureCanvas and sizes the window for comfortable inspection of spectral loadings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object containing the PCA loading plot to display.
    parent : QWidget, optional
        Parent widget that will own this window, by default None.
    """

    def __init__(self, fig, parent=None):
        super().__init__(parent)
        QTimer.singleShot(0, lambda: retranslate_widget_tree(self, get_language()))
        self.setWindowTitle(tr("PCA Loading Plot"))
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        self.setLayout(layout)
        self.show()