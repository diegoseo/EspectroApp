import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters as exporters
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


from core.translations import translate, get_language


def tr(text, **values):
    return translate(text, get_language(), **values)


class SpectraPlotWindow(QWidget):
    def __init__(
        self,
        datos,
        raman_shift,
        asignacion_colores,
        x_label="X Axis",
        y_label="Intensity",
    ):
        """
        Displays multiple spectra in a single interactive plotting window.
        The window groups spectra by type, overlays individual curves, and highlights the mean spectrum per class.

        Parameters
        ----------
        datos : pandas.DataFrame
            DataFrame in the internal format where the first column is the X-axis and the first row after it
            contains sample types or classes, followed by intensity values.
        raman_shift : array-like
            X-axis values (e.g., Raman shift) corresponding to the spectral measurements.
        asignacion_colores : dict
            Mapping from sample type or class name to a color representation understood by pyqtgraph.
        x_label : str, optional
            Label for the X-axis displayed on the plot, by default "X Axis".
        y_label : str, optional
            Label for the Y-axis displayed on the plot, by default "Intensity".
        """
        super().__init__()
        self.setWindowTitle(tr("Spectra Plot"))
        self.resize(1200, 650)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.btn_save = QPushButton(tr("Save high-resolution image"))
        self.btn_save.clicked.connect(self.save_high_resolution_image)
        layout.addWidget(self.btn_save)

        apply_nature_style_pg(self.plot_widget, x_label, y_label)

        datos = datos.iloc[:, 1:]

        tipos = datos.iloc[0, :]
        intensidades = datos.iloc[1:, :].copy()

        intensidades.columns = tipos.values
        intensidades = intensidades.astype(float)
        datos = intensidades

        tipos_unicos = datos.columns.unique()
        x = np.array(raman_shift, dtype=float)

        self.legend = pg.LegendItem(labelTextSize="14pt", labelTextColor="k")
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.anchor((1, 0), (1, 0), offset=(-12, 10))

        for tipo in tipos_unicos:
            indices = [i for i, col in enumerate(datos.columns) if col == tipo]

            color_actual = asignacion_colores.get(tipo, "#000000")

            pen_individual = pg.mkPen(
                apply_color_alpha(color_actual, alpha=35), width=0.35
            )

            matriz_tipo = []

            for idx in indices:
                y_fila = datos.iloc[:, idx]

                if isinstance(y_fila, pd.DataFrame):
                    y_fila = y_fila.iloc[:, 0]

                try:
                    y = np.array(y_fila, dtype=float).flatten()
                    matriz_tipo.append(y)
                    self.plot_widget.plot(x, y, pen=pen_individual)

                except Exception as e:
                    print(f"Error plotting column {idx} ({tipo}): {e}")

            if matriz_tipo:
                matriz_tipo = np.vstack(matriz_tipo)
                y_promedio = np.nanmean(matriz_tipo, axis=0)

                pen_promedio = pg.mkPen(pg.mkColor(color_actual), width=2.5)
                curva_promedio = self.plot_widget.plot(
                    x, y_promedio, pen=pen_promedio, name=str(tipo)
                )

                self.legend.addItem(curva_promedio, str(tipo))

    def save_high_resolution_image(self):
        """
        Saves the current spectra plot as a high-resolution image file.
        The user can choose between PNG and SVG formats and the plot is exported with publication-quality resolution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ruta, _ = QFileDialog.getSaveFileName(
            self, tr("Save plot"), "spectra_high_resolution.png", tr("PNG (*.png);;SVG (*.svg)")
        )

        if not ruta:
            return

        try:
            if ruta.lower().endswith(".svg"):
                exporter = exporters.SVGExporter(self.plot_widget.plotItem)
                exporter.export(ruta)
            else:
                if not ruta.lower().endswith(".png"):
                    ruta += ".png"

                exporter = exporters.ImageExporter(self.plot_widget.plotItem)

                exporter.parameters()["width"] = 5000

                exporter.export(ruta)

            QMessageBox.information(
                self,
                tr("Success"),
                tr("Plot saved to:\n{path}", path=ruta),
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                tr("Error"),
                tr("The plot could not be saved:\n{error}", error=e),
            )


class StackedSpectraPlotWindow(QWidget):
    """Display spectra with an adjustable vertical offset."""

    def __init__(
        self,
        datos,
        raman_shift,
        asignacion_colores,
        x_label="X Axis",
        y_label="Intensity",
        offset_mode="automatic",
        offset_value=1.15,
        show_labels=True,
        maximum_spectra=10,
        sample_type=None,
        range_min=None,
        range_max=None,
    ):
        super().__init__()

        self.setWindowTitle(tr("Stacked Spectra"))
        self.resize(1250, 720)

        self.figure = plt.Figure(figsize=(12, 7))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(
            self.canvas,
            self,
        )

        self.axes = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.btn_save = QPushButton(tr("Save high-resolution image"))
        self.btn_save.clicked.connect(self.save_high_resolution_image)

        button_layout.addWidget(self.btn_save)
        layout.addLayout(button_layout)

        self._plot_stacked_spectra(
            datos=datos,
            raman_shift=raman_shift,
            asignacion_colores=asignacion_colores,
            x_label=x_label,
            y_label=y_label,
            offset_mode=offset_mode,
            offset_value=offset_value,
            show_labels=show_labels,
            maximum_spectra=maximum_spectra,
            sample_type=sample_type,
            range_min=range_min,
            range_max=range_max,
        )

    def _plot_stacked_spectra(
        self,
        datos,
        raman_shift,
        asignacion_colores,
        x_label,
        y_label,
        offset_mode,
        offset_value,
        show_labels,
        maximum_spectra,
        sample_type,
        range_min,
        range_max,
    ):
        dataframe = datos.copy()

        sample_types = dataframe.iloc[0, 1:].astype(str).tolist()

        x_values = pd.to_numeric(
            pd.Series(raman_shift),
            errors="coerce",
        ).to_numpy(dtype=float)

        spectra = dataframe.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")

        valid_x = np.isfinite(x_values)
        x_values = x_values[valid_x]
        spectra = spectra.loc[valid_x].reset_index(drop=True)

        if range_min is not None and range_max is not None:
            range_mask = (x_values >= float(range_min)) & (x_values <= float(range_max))
            x_values = x_values[range_mask]
            spectra = spectra.loc[range_mask].reset_index(drop=True)

        selected_columns = [
            index
            for index, current_type in enumerate(sample_types)
            if (sample_type is None or str(current_type) == str(sample_type))
        ]

        if not selected_columns:
            raise ValueError(tr("No spectra match the selected sample type."))

        maximum_spectra = max(
            1,
            int(maximum_spectra),
        )

        if len(selected_columns) > maximum_spectra:
            sampled_positions = np.linspace(
                0,
                len(selected_columns) - 1,
                maximum_spectra,
                dtype=int,
            )

            selected_columns = [
                selected_columns[position] for position in sampled_positions
            ]

        spectra_arrays = []
        labels = []
        colors = []

        type_counters = {}

        for column_index in selected_columns:
            y_values = spectra.iloc[
                :,
                column_index,
            ].to_numpy(dtype=float)

            if not np.isfinite(y_values).any():
                continue

            current_type = sample_types[column_index]
            type_counters[current_type] = type_counters.get(current_type, 0) + 1

            spectra_arrays.append(y_values)
            labels.append(f"{current_type} {type_counters[current_type]}")
            colors.append(
                asignacion_colores.get(
                    current_type,
                    "#000000",
                )
            )

        if not spectra_arrays:
            raise ValueError(tr("The selected matrix has no numeric spectra."))

        robust_ranges = []

        for y_values in spectra_arrays:
            finite_values = y_values[np.isfinite(y_values)]

            if finite_values.size:
                robust_range = np.nanpercentile(finite_values, 95) - np.nanpercentile(
                    finite_values, 5
                )

                if robust_range > 0:
                    robust_ranges.append(robust_range)

        base_amplitude = float(np.nanmedian(robust_ranges)) if robust_ranges else 1.0

        if offset_mode == "manual":
            vertical_step = float(offset_value)
        else:
            vertical_step = base_amplitude * float(offset_value)

        if vertical_step <= 0:
            vertical_step = base_amplitude or 1.0

        x_span = (
            float(np.nanmax(x_values) - np.nanmin(x_values)) if x_values.size else 1.0
        )
        label_x = float(np.nanmax(x_values)) + x_span * 0.018

        for spectrum_index, (
            y_values,
            label,
            color,
        ) in enumerate(
            zip(
                spectra_arrays,
                labels,
                colors,
            )
        ):
            offset = spectrum_index * vertical_step
            shifted_values = y_values + offset

            self.axes.plot(
                x_values,
                shifted_values,
                linewidth=1.15,
                color=color,
            )

            if show_labels:
                finite_values = shifted_values[np.isfinite(shifted_values)]

                if finite_values.size:
                    label_y = float(
                        np.nanmedian(
                            finite_values[
                                -max(
                                    1,
                                    finite_values.size // 20,
                                ) :
                            ]
                        )
                    )

                    self.axes.text(
                        label_x,
                        label_y,
                        label,
                        color=color,
                        fontsize=9,
                        va="center",
                        clip_on=False,
                    )

        self.axes.set_title(
            tr("Stacked spectra"),
            fontsize=15,
            fontweight="bold",
            pad=12,
        )
        self.axes.set_xlabel(x_label, fontsize=12)
        self.axes.set_ylabel(
            tr("{label} + vertical offset", label=y_label),
            fontsize=12,
        )
        self.axes.grid(False)
        self.axes.tick_params(
            direction="out",
            labelsize=10,
        )
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)

        if show_labels:
            self.figure.subplots_adjust(
                right=0.80,
                left=0.10,
                bottom=0.12,
                top=0.90,
            )
        else:
            self.figure.tight_layout()

        self.canvas.draw_idle()

    def save_high_resolution_image(self):
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            tr("Save stacked spectra"),
            "stacked_spectra.png",
            tr(
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
                tr("Success"),
                tr("Plot saved to:\n{path}", path=path),
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                tr("Error"),
                tr("The plot could not be saved:\n{error}", error=error),
            )


class LimitedRangeSpectraPlotWindow(QWidget):
    def __init__(
        self,
        datos,
        raman_shift,
        asignacion_colores,
        val_min,
        val_max,
        x_label="X Axis",
        y_label="Intensity",
    ):
        """
        Displays spectra within a restricted X-axis range in an interactive plotting window.
        The window filters each spectrum to the selected interval, overlays individual curves, and emphasizes the mean spectrum for each class.

        Parameters
        ----------
        datos : pandas.DataFrame
            DataFrame in the internal format where the first column is the X-axis and the first row after it
            contains sample types or classes, followed by intensity values.
        raman_shift : array-like
            Full X-axis values (e.g., Raman shift) corresponding to the spectral measurements.
        asignacion_colores : dict
            Mapping from sample type or class name to a color representation understood by pyqtgraph.
        val_min : float
            Lower bound of the X-axis range to display.
        val_max : float
            Upper bound of the X-axis range to display.
        x_label : str, optional
            Label for the X-axis displayed on the plot, by default "X Axis".
        y_label : str, optional
            Label for the Y-axis displayed on the plot, by default "Intensity".
        """

        super().__init__()
        self.setWindowTitle(tr("Limited-Range Plot"))
        self.resize(1200, 650)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.btn_save = QPushButton(tr("Save high-resolution image"))
        self.btn_save.clicked.connect(self.save_high_resolution_image)
        layout.addWidget(self.btn_save)

        apply_nature_style_pg(self.plot_widget, x_label, y_label)

        self.legend = pg.LegendItem(labelTextSize="14pt", labelTextColor="k")
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.anchor((1, 0), (1, 0), offset=(-12, 10))

        datos = datos.iloc[:, 1:]

        tipos = datos.iloc[0, :]

        intensidades = datos.iloc[1:, :].copy()
        intensidades.columns = tipos.values
        intensidades = intensidades.apply(pd.to_numeric, errors="coerce")

        datos = intensidades

        tipos_unicos = datos.columns.unique()

        x_total = np.array(raman_shift, dtype=float).flatten()

        mascara = (x_total >= val_min) & (x_total <= val_max)
        x_filtrado = x_total[mascara]

        for tipo in tipos_unicos:
            indices = [i for i, col in enumerate(datos.columns) if col == tipo]

            color_actual = asignacion_colores.get(tipo, "#000000")

            color_transparente = pg.mkColor(color_actual)
            color_transparente.setAlpha(35)
            pen_individual = pg.mkPen(color_transparente, width=0.35)

            matriz_tipo = []

            for idx in indices:
                y_fila = datos.iloc[:, idx]

                if isinstance(y_fila, pd.DataFrame):
                    y_fila = y_fila.iloc[:, 0]

                try:
                    y_total = np.array(y_fila, dtype=float).flatten()
                    y_filtrado = y_total[mascara]

                    matriz_tipo.append(y_filtrado)

                    self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen_individual)

                except Exception as e:
                    print(f"Error plotting column {idx} ({tipo}): {e}")

            if matriz_tipo:
                matriz_tipo = np.vstack(matriz_tipo)
                y_promedio = np.nanmean(matriz_tipo, axis=0)

                pen_promedio = pg.mkPen(pg.mkColor(color_actual), width=2.5)

                curva_promedio = self.plot_widget.plot(
                    x_filtrado, y_promedio, pen=pen_promedio, name=str(tipo)
                )

                self.legend.addItem(curva_promedio, str(tipo))

    def save_high_resolution_image(self):
        """
        Saves the current limited-range spectra plot as a high-resolution image file.
        The user can choose between PNG and SVG formats and the plot is exported with publication-quality resolution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ruta, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            "limited_range_spectra_high_resolution.png",
            "PNG (*.png);;SVG (*.svg)",
        )

        if not ruta:
            return

        try:
            if ruta.lower().endswith(".svg"):
                exporter = exporters.SVGExporter(self.plot_widget.plotItem)
                exporter.export(ruta)

            else:
                if not ruta.lower().endswith(".png"):
                    ruta += ".png"

                exporter = exporters.ImageExporter(self.plot_widget.plotItem)

                exporter.parameters()["width"] = 5000

                exporter.export(ruta)

            QMessageBox.information(
                self,
                tr("Success"),
                tr("Plot saved to:\n{path}", path=ruta),
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                tr("Error"),
                tr("The plot could not be saved:\n{error}", error=e),
            )


class SpectraByTypePlotWindow(QWidget):
    def __init__(
        self,
        datos,
        raman_shift,
        asignacion_colores,
        tipo_deseado,
        x_label="X Axis",
        y_label="Intensity",
    ):
        """
        Displays spectra of a single selected type in an interactive plotting window.
        The window overlays all individual spectra of the chosen class and highlights their mean spectrum for easier comparison.

        Parameters
        ----------
        datos : pandas.DataFrame
            DataFrame in the internal format where the first column is the X-axis and the first row after it
            contains sample types or classes, followed by intensity values.
        raman_shift : array-like
            X-axis values (e.g., Raman shift) corresponding to the spectral measurements.
        asignacion_colores : dict
            Mapping from sample type or class name to a color representation understood by pyqtgraph.
        tipo_deseado : str
            Name of the sample type or class to be displayed in the plot.
        x_label : str, optional
            Label for the X-axis displayed on the plot, by default "X Axis".
        y_label : str, optional
            Label for the Y-axis displayed on the plot, by default "Intensity".
        """
        super().__init__()
        self.setWindowTitle(tr("Spectra Plot by Type"))
        self.resize(1200, 650)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.btn_save = QPushButton(tr("Save high-resolution image"))
        self.btn_save.clicked.connect(self.save_high_resolution_image)
        layout.addWidget(self.btn_save)

        apply_nature_style_pg(self.plot_widget, x_label, y_label)

        self.legend = pg.LegendItem(labelTextSize="14pt", labelTextColor="k")
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.anchor((1, 0), (1, 0), offset=(-12, 10))

        datos = datos.iloc[:, 1:]

        tipos = datos.iloc[0, :]

        intensidades = datos.iloc[1:, :].copy()
        intensidades.columns = tipos.values
        intensidades = intensidades.apply(pd.to_numeric, errors="coerce")

        datos = intensidades

        x = np.array(raman_shift, dtype=float).flatten()

        indices = [i for i, col in enumerate(datos.columns) if col == tipo_deseado]

        color_actual = asignacion_colores.get(tipo_deseado, "#000000")

        color_transparente = pg.mkColor(color_actual)
        color_transparente.setAlpha(35)
        pen_individual = pg.mkPen(color_transparente, width=0.35)

        matriz_tipo = []

        for idx in indices:
            y_fila = datos.iloc[:, idx]

            if isinstance(y_fila, pd.DataFrame):
                y_fila = y_fila.iloc[:, 0]

            try:
                y = np.array(y_fila, dtype=float).flatten()
                matriz_tipo.append(y)

                self.plot_widget.plot(x, y, pen=pen_individual)

            except Exception as e:
                print(f"Error plotting column {idx} ({tipo_deseado}): {e}")

        if matriz_tipo:
            matriz_tipo = np.vstack(matriz_tipo)
            y_promedio = np.nanmean(matriz_tipo, axis=0)

            pen_promedio = pg.mkPen(pg.mkColor(color_actual), width=2.5)

            curva_promedio = self.plot_widget.plot(
                x, y_promedio, pen=pen_promedio, name=str(tipo_deseado)
            )

            self.legend.addItem(curva_promedio, str(tipo_deseado))

    def save_high_resolution_image(self):
        """
        Saves the current spectra-by-type plot as a high-resolution image file.
        The user can choose between PNG and SVG formats and the plot is exported with publication-quality resolution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ruta, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            "spectra_by_type_high_resolution.png",
            "PNG (*.png);;SVG (*.svg)",
        )

        if not ruta:
            return

        try:
            if ruta.lower().endswith(".svg"):
                exporter = exporters.SVGExporter(self.plot_widget.plotItem)
                exporter.export(ruta)

            else:
                if not ruta.lower().endswith(".png"):
                    ruta += ".png"

                exporter = exporters.ImageExporter(self.plot_widget.plotItem)

                exporter.parameters()["width"] = 5000

                exporter.export(ruta)

            QMessageBox.information(
                self,
                tr("Success"),
                tr("Plot saved to:\n{path}", path=ruta),
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                tr("Error"),
                tr("The plot could not be saved:\n{error}", error=e),
            )


class LimitedRangeSpectraByTypePlotWindow(QWidget):
    def __init__(
        self,
        datos,
        raman_shift,
        asignacion_colores,
        tipo_deseado,
        val_min,
        val_max,
        x_label="X Axis",
        y_label="Intensity",
    ):
        """
        Displays spectra of a single selected type within a restricted X-axis range in an interactive plotting window.
        The window filters the chosen class of spectra to the specified interval, overlays individual curves, and emphasizes their mean spectrum.

        Parameters
        ----------
        datos : pandas.DataFrame
            DataFrame in the internal format where the first column is the X-axis and the first row after it
            contains sample types or classes, followed by intensity values.
        raman_shift : array-like
            Full X-axis values (e.g., Raman shift) corresponding to the spectral measurements.
        asignacion_colores : dict
            Mapping from sample type or class name to a color representation understood by pyqtgraph.
        tipo_deseado : str
            Name of the sample type or class to be displayed in the plot.
        val_min : float
            Lower bound of the X-axis range to display.
        val_max : float
            Upper bound of the X-axis range to display.
        x_label : str, optional
            Label for the X-axis displayed on the plot, by default "X Axis".
        y_label : str, optional
            Label for the Y-axis displayed on the plot, by default "Intensity".
        """

        super().__init__()
        self.setWindowTitle(tr("Limited-Range Plot by Type"))
        self.resize(1200, 650)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.btn_save = QPushButton(tr("Save high-resolution image"))
        self.btn_save.clicked.connect(self.save_high_resolution_image)
        layout.addWidget(self.btn_save)

        apply_nature_style_pg(self.plot_widget, x_label, y_label)

        self.legend = pg.LegendItem(labelTextSize="14pt", labelTextColor="k")
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.anchor((1, 0), (1, 0), offset=(-12, 10))

        datos = datos.iloc[:, 1:]

        tipos = datos.iloc[0, :]

        intensidades = datos.iloc[1:, :].copy()
        intensidades.columns = tipos.values
        intensidades = intensidades.apply(pd.to_numeric, errors="coerce")

        datos = intensidades

        x_total = np.array(raman_shift, dtype=float).flatten()

        mascara = (x_total >= val_min) & (x_total <= val_max)
        x_filtrado = x_total[mascara]

        indices = [i for i, col in enumerate(datos.columns) if col == tipo_deseado]

        color_actual = asignacion_colores.get(tipo_deseado, "#000000")

        color_transparente = pg.mkColor(color_actual)
        color_transparente.setAlpha(35)
        pen_individual = pg.mkPen(color_transparente, width=0.35)

        matriz_tipo = []

        for idx in indices:
            y_fila = datos.iloc[:, idx]

            if isinstance(y_fila, pd.DataFrame):
                y_fila = y_fila.iloc[:, 0]

            try:
                y_total = np.array(y_fila, dtype=float).flatten()

                y_filtrado = y_total[mascara]

                matriz_tipo.append(y_filtrado)

                self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen_individual)

            except Exception as e:
                print(f"Error plotting column {idx} ({tipo_deseado}): {e}")

        if matriz_tipo:
            matriz_tipo = np.vstack(matriz_tipo)
            y_promedio = np.nanmean(matriz_tipo, axis=0)

            pen_promedio = pg.mkPen(pg.mkColor(color_actual), width=2.5)

            curva_promedio = self.plot_widget.plot(
                x_filtrado, y_promedio, pen=pen_promedio, name=str(tipo_deseado)
            )

            self.legend.addItem(curva_promedio, str(tipo_deseado))

    def save_high_resolution_image(self):
        """
        Saves the current limited-range spectra-by-type plot as a high-resolution image file.
        The user can choose between PNG and SVG formats and the plot is exported with publication-quality resolution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ruta, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            "limited_range_spectra_by_type_high_resolution.png",
            "PNG (*.png);;SVG (*.svg)",
        )

        if not ruta:
            return

        try:
            if ruta.lower().endswith(".svg"):
                exporter = exporters.SVGExporter(self.plot_widget.plotItem)
                exporter.export(ruta)

            else:
                if not ruta.lower().endswith(".png"):
                    ruta += ".png"

                exporter = exporters.ImageExporter(self.plot_widget.plotItem)

                exporter.parameters()["width"] = 5000

                exporter.export(ruta)

            QMessageBox.information(
                self,
                tr("Success"),
                tr("Plot saved to:\n{path}", path=ruta),
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                tr("Error"),
                tr("The plot could not be saved:\n{error}", error=e),
            )


from sklearn.model_selection import StratifiedKFold, cross_val_score


def calculate_accuracy(dataframe_pca, etiquetas):
    """
    Calcula la exactitud de clasificación mediante KNN con k=3
    y validación cruzada estratificada.

    Se utilizan 5 folds cuando todas las clases tienen al menos
    5 muestras. Si la clase minoritaria tiene menos de 5 muestras,
    el número de folds se reduce automáticamente.

    Parameters
    ----------
    dataframe_pca : pandas.DataFrame
        DataFrame que contiene las coordenadas numéricas del PCA
        o del t-SNE.

    etiquetas : array-like
        Etiquetas de clase correspondientes a cada muestra.

    Returns
    -------
    float
        Exactitud media expresada en porcentaje.
        Devuelve 0.0 cuando no es posible realizar una validación
        cruzada estratificada válida.
    """
    columnas_numericas = [
        columna
        for columna in dataframe_pca.columns
        if pd.api.types.is_numeric_dtype(dataframe_pca[columna])
    ]

    if not columnas_numericas:
        print("[ACCURACY] No se encontraron columnas numéricas.")
        return 0.0

    X = dataframe_pca[columnas_numericas].copy().reset_index(drop=True)

    y = pd.Series(etiquetas).reset_index(drop=True)

    df_completo = X.copy()
    df_completo["__etiqueta__"] = y

    df_completo = df_completo.dropna()

    if df_completo.empty:
        print(
            "[ACCURACY] No existen datos válidos "
            "después de eliminar valores faltantes."
        )
        return 0.0

    X_clean = df_completo.drop(columns=["__etiqueta__"]).to_numpy(dtype=float)

    y_clean = df_completo["__etiqueta__"].to_numpy()

    if len(X_clean) < 4:
        print(
            "[ACCURACY] No existen suficientes muestras " "para utilizar KNN con k=3."
        )
        return 0.0

    conteos_clase = pd.Series(y_clean).value_counts()

    print("[ACCURACY] Muestras por clase:")
    print(conteos_clase)

    if len(conteos_clase) < 2:
        print("[ACCURACY] Se necesita más de una clase " "para calcular la exactitud.")
        return 0.0

    minimo_por_clase = int(conteos_clase.min())

    if minimo_por_clase < 2:
        print(
            "[ACCURACY] Existe una clase con menos de "
            "dos muestras. No se puede aplicar "
            "validación cruzada estratificada."
        )
        return 0.0

    n_splits = min(
        5,
        minimo_por_clase,
    )

    n_vecinos = 3

    tamaño_test_maximo = int(np.ceil(len(X_clean) / n_splits))

    tamaño_train_minimo = len(X_clean) - tamaño_test_maximo

    if tamaño_train_minimo < n_vecinos:
        print(
            "[ACCURACY] El conjunto de entrenamiento "
            "es demasiado pequeño para utilizar KNN con k=3."
        )
        return 0.0

    print(
        "[ACCURACY] Configuración:",
        f"folds={n_splits},",
        f"k={n_vecinos}",
    )

    clasificador = KNeighborsClassifier(n_neighbors=n_vecinos)

    validacion = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        clasificador,
        X_clean,
        y_clean,
        cv=validacion,
        scoring="accuracy",
    )

    exactitud = scores.mean() * 100

    return round(
        float(exactitud),
        2,
    )


def calculate_accuracyviejo(dataframe_pca, etiquetas):
    """
    Estimates the classification accuracy achievable from PCA-transformed data using a simple k-NN model.
    The function filters valid numeric rows, performs a train/test split, and reports the percentage of correctly classified samples.

    Parameters
    ----------
    dataframe_pca : pandas.DataFrame
        DataFrame containing PCA components and possibly other columns; only numeric columns are used as features.
    etiquetas : array-like
        Class labels corresponding to the rows of the PCA DataFrame.

    Returns
    -------
    float
        Estimated classification accuracy expressed as a percentage, or 0 if there is insufficient data to train the model.
    """
    columnas_numericas = [
        col
        for col in dataframe_pca.columns
        if dataframe_pca[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    X = dataframe_pca[columnas_numericas]
    y = etiquetas

    df_completo = X.copy()
    df_completo["__etiqueta__"] = y
    df_completo = df_completo.dropna()

    X_clean = df_completo.drop(columns=["__etiqueta__"]).values
    y_clean = df_completo["__etiqueta__"].values

    if len(X_clean) < 3:
        return 0

    X_scaled = X_clean

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_clean, test_size=0.3, random_state=42
    )

    n_vecinos = min(3, len(X_train))

    if n_vecinos < 1:
        return 0

    clf = KNeighborsClassifier(n_neighbors=n_vecinos)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return round(accuracy * 100, 2)


def graficar_varianza_acumulada(
    var_acum, var_ind=None, umbral=95, max_cp=20, anotar=True
):
    """
    Creates a bar/line plot showing individual and cumulative explained variance from a PCA analysis.
    The figure highlights the number of components needed to reach a chosen variance threshold and optionally annotates percentages.

    Parameters
    ----------
    var_acum : array-like
        Cumulative explained variance values in percentage for each principal component, ordered from first to last.
    var_ind : array-like, optional
        Individual explained variance values in percentage for each principal component, used for the bar plot if provided.
    umbral : float, optional
        Target cumulative variance threshold (in percent) to highlight with a horizontal line, by default 95.
    max_cp : int, optional
        Maximum number of principal components to display on the plot, by default 20.
    anotar : bool, optional
        If True, annotates each cumulative variance point with its percentage value, by default True.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object containing the variance plot, ready for display or saving.
    """
    n_total = len(var_acum)
    n = min(max_cp, n_total)

    comps = np.arange(1, n + 1)
    acum = var_acum[:n]

    if var_ind is not None:
        ind = var_ind[:n]
    else:
        ind = None

    n_umbral = int(np.argmax(var_acum >= umbral) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    if ind is not None:
        ax.bar(comps, ind, alpha=0.75, label="Individual")

    ax.plot(comps, acum, marker="o", color="black", label="Cumulative")

    ax.axhline(umbral, color="red", linestyle="--", label=f"{umbral}%")

    if n_umbral <= n:
        ax.axvline(n_umbral, color="green", linestyle="--", label=f"{n_umbral} PCs")

    ax.set_xlabel(tr("Principal Component"))
    ax.set_ylabel(tr("Explained variance (%)"))
    ax.set_title(tr("Cumulative explained variance (PCA)"))
    ax.set_ylim(0, 105)
    ax.set_xticks(comps)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    if anotar:
        for x, y_acum in zip(comps, acum):
            ax.text(x, y_acum + 1.0, f"{y_acum:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    return fig


def apply_color_alpha(color, alpha=60):
    """
    Creates a semi-transparent version of a given color for use in pyqtgraph plots.
    The function adjusts the alpha channel of the input color while keeping its original hue.

    Parameters
    ----------
    color : any
        Color specification understood by pyqtgraph (e.g., name, hex string, or QColor).
    alpha : int, optional
        Alpha channel value between 0 (fully transparent) and 255 (fully opaque), by default 60.

    Returns
    -------
    QColor
        QColor instance with the requested transparency applied.
    """
    qcolor = pg.mkColor(color)
    qcolor.setAlpha(alpha)
    return qcolor


def apply_nature_style_pg(plot_widget, etiqueta_x, etiqueta_y):
    """
    Applies a journal-style visual theme to a pyqtgraph plot widget.
    The function configures axes, fonts, colors, and margins so spectra plots resemble publication-quality figures.

    Parameters
    ----------
    plot_widget : pyqtgraph.PlotWidget
        Target plot widget on which the visual style will be applied.
    etiqueta_x : str
        Text label to use for the X-axis.
    etiqueta_y : str
        Text label to use for the Y-axis.

    Returns
    -------
    None
    """
    plot_widget.setBackground("w")
    plot_widget.showGrid(x=False, y=False)

    plot_widget.setMenuEnabled(True)
    plot_widget.getPlotItem().hideButtons()

    plot_widget.getPlotItem().setContentsMargins(15, 12, 12, 15)

    for axis_name in ["left", "bottom"]:
        axis = plot_widget.getAxis(axis_name)

        axis.setPen(pg.mkPen("k", width=1.2))
        axis.setTextPen("k")
        axis.enableAutoSIPrefix(False)

        axis.setStyle(tickFont=QFont("Arial", 11), tickLength=-5, showValues=True)

    plot_widget.getAxis("left").setWidth(75)
    plot_widget.getAxis("bottom").setHeight(55)

    plot_widget.setLabel(
        "left", etiqueta_y, color="k", **{"font-size": "13pt", "font-family": "Arial"}
    )

    plot_widget.setLabel(
        "bottom", etiqueta_x, color="k", **{"font-size": "13pt", "font-family": "Arial"}
    )