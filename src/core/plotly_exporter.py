"""Export Plotly figures through Qt WebEngine without Kaleido or Chrome."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, QSize, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from core.translations import translate, get_language


def tr(text, **values):
    return translate(text, get_language(), **values)


class PlotlyExporter(QObject):
    """
    Export Plotly figures already rendered in a QWebEngineView.

    PNG is captured from the actual Qt WebEngine rendering, avoiding the blank
    images produced when QtSvg tries to rasterize Plotly's CSS-dependent SVG.
    SVG is serialized from the rendered Plotly DOM with its relevant style
    rules embedded. PDF uses Qt WebEngine's native printToPdf callback.
    """

    export_finished = Signal(str)
    export_error = Signal(str)

    def __init__(self, web_view, parent=None):
        super().__init__(parent)
        self.web_view = web_view
        self._destination = ""
        self._image_format = ""
        self._width = 1600
        self._height = 1000
        self._scale = 2
        self._original_size = QSize()

    def export_image(
        self,
        file_path: str,
        image_format: str,
        width: int = 1600,
        height: int = 1000,
        scale: int = 2,
    ) -> None:
        """Export the rendered Plotly figure as PNG or SVG."""
        image_format = image_format.lower().strip()

        if image_format not in {"png", "svg"}:
            self.export_error.emit(
                tr(
                    "Unsupported Plotly image format: {format}",
                    format=image_format,
                )
            )
            return

        if self.web_view is None or self.web_view.page() is None:
            self.export_error.emit(tr("The Plotly web view is not available."))
            return

        self._destination = str(Path(file_path).expanduser().resolve())
        self._image_format = image_format
        self._width = max(1, int(width))
        self._height = max(1, int(height))
        self._scale = max(1, int(scale))

        if image_format == "png":
            self._export_png_from_web_view()
        else:
            self._export_svg_from_dom()

    def _export_png_from_web_view(self) -> None:
        """
        Capture the actual QWebEngineView at the requested pixel dimensions.

        This is not a desktop screenshot: Qt grabs the rendered web widget
        directly. The view is temporarily resized, Plotly is told to resize,
        and the original widget size is restored afterwards.
        """
        self._original_size = self.web_view.size()

        target_width = self._width * self._scale
        target_height = self._height * self._scale

        self.web_view.resize(target_width, target_height)
        self.web_view.updateGeometry()
        self.web_view.update()

        javascript = """
        (() => {
            const graph = document.querySelector(".plotly-graph-div");
            if (graph && typeof Plotly !== "undefined") {
                Plotly.Plots.resize(graph);
            }
            return true;
        })()
        """

        self.web_view.page().runJavaScript(
            javascript,
            lambda _result: QTimer.singleShot(
                700,
                self._capture_png,
            ),
        )

    def _capture_png(self) -> None:
        try:
            QApplication.processEvents()
            pixmap = self.web_view.grab()

            if pixmap.isNull():
                raise ValueError(tr("Qt WebEngine returned an empty image."))

            destination = Path(self._destination)
            destination.parent.mkdir(parents=True, exist_ok=True)

            image = pixmap.toImage()
            if image.format() == QImage.Format_Invalid:
                raise ValueError(tr("Qt WebEngine returned an invalid image."))

            if not image.save(str(destination), "PNG"):
                raise OSError(tr("The PNG file could not be written."))

        except (OSError, ValueError) as error:
            self._restore_web_view_size()
            self.export_error.emit(str(error))
            return

        self._restore_web_view_size()
        self.export_finished.emit(os.path.abspath(self._destination))

    def _restore_web_view_size(self) -> None:
        if self._original_size.isValid():
            self.web_view.resize(self._original_size)
            self.web_view.updateGeometry()
            self.web_view.update()

            self.web_view.page().runJavaScript("""
                (() => {
                    const graph = document.querySelector(".plotly-graph-div");
                    if (graph && typeof Plotly !== "undefined") {
                        Plotly.Plots.resize(graph);
                    }
                    return true;
                })()
                """)

    def _export_svg_from_dom(self) -> None:
        """
        Serialize all Plotly SVG layers and embed page CSS.

        Plotly separates the chart, legend and overlays into several SVG
        elements. Exporting only one SVG omits parts of the figure.
        """
        javascript = f"""
        (() => {{
            try {{
                const graph = document.querySelector(".plotly-graph-div");
                if (!graph) {{
                    return JSON.stringify({{
                        ok: false,
                        error: "The Plotly graph was not found."
                    }});
                }}

                const container = graph.querySelector(".svg-container");
                if (!container) {{
                    return JSON.stringify({{
                        ok: false,
                        error: "The Plotly SVG container was not found."
                    }});
                }}

                const layers = Array.from(
                    container.querySelectorAll(":scope > svg")
                );

                if (!layers.length) {{
                    return JSON.stringify({{
                        ok: false,
                        error: (
                            "This figure has no SVG layers. "
                            + "Use PDF or HTML for WebGL figures."
                        )
                    }});
                }}

                const sourceWidth = graph.clientWidth || 800;
                const sourceHeight = graph.clientHeight || 600;
                const ns = "http://www.w3.org/2000/svg";

                const root = document.createElementNS(ns, "svg");
                root.setAttribute("xmlns", ns);
                root.setAttribute(
                    "xmlns:xlink",
                    "http://www.w3.org/1999/xlink"
                );
                root.setAttribute("width", "{self._width}");
                root.setAttribute("height", "{self._height}");
                root.setAttribute(
                    "viewBox",
                    `0 0 ${{sourceWidth}} ${{sourceHeight}}`
                );
                root.setAttribute(
                    "preserveAspectRatio",
                    "xMidYMid meet"
                );

                const style = document.createElementNS(ns, "style");
                let cssText = "";

                for (const sheet of Array.from(document.styleSheets)) {{
                    try {{
                        for (const rule of Array.from(sheet.cssRules || [])) {{
                            cssText += rule.cssText + "\\n";
                        }}
                    }} catch (_error) {{
                        // Ignore inaccessible stylesheets.
                    }}
                }}

                style.textContent = cssText;
                root.appendChild(style);

                layers.forEach((layer) => {{
                    const clone = layer.cloneNode(true);
                    clone.removeAttribute("style");
                    clone.setAttribute("x", "0");
                    clone.setAttribute("y", "0");
                    clone.setAttribute("width", String(sourceWidth));
                    clone.setAttribute("height", String(sourceHeight));
                    root.appendChild(clone);
                }});

                return JSON.stringify({{
                    ok: true,
                    svg: new XMLSerializer().serializeToString(root)
                }});

            }} catch (error) {{
                return JSON.stringify({{
                    ok: false,
                    error: String(error)
                }});
            }}
        }})()
        """

        self.web_view.page().runJavaScript(
            javascript,
            self._handle_svg_result,
        )

    def _handle_svg_result(self, result) -> None:
        if not isinstance(result, str) or not result.strip():
            self.export_error.emit("Qt WebEngine did not return valid SVG text.")
            return

        try:
            payload = json.loads(result)
        except json.JSONDecodeError as error:
            self.export_error.emit(
                tr(
                    "Qt WebEngine returned invalid JSON: {error}",
                    error=error,
                )
            )
            return

        if not isinstance(payload, dict):
            self.export_error.emit("Qt WebEngine returned an unexpected SVG response.")
            return

        if not payload.get("ok"):
            self.export_error.emit(
                str(payload.get("error") or "The SVG could not be extracted.")
            )
            return

        svg_text = payload.get("svg")
        if not isinstance(svg_text, str) or "<svg" not in svg_text:
            self.export_error.emit("The extracted SVG is invalid.")
            return

        try:
            destination = Path(self._destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(svg_text, encoding="utf-8")
        except OSError as error:
            self.export_error.emit(str(error))
            return

        self.export_finished.emit(os.path.abspath(self._destination))

    def export_pdf(self, file_path: str) -> None:
        """Export the current Plotly page as PDF bytes."""
        if self.web_view is None or self.web_view.page() is None:
            self.export_error.emit(tr("The Plotly web view is not available."))
            return

        destination = str(Path(file_path).expanduser().resolve())

        def pdf_ready(pdf_data) -> None:
            try:
                raw_data = bytes(pdf_data or b"")
                if not raw_data:
                    raise ValueError(tr("Qt WebEngine returned an empty PDF."))

                output_path = Path(destination)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(raw_data)

            except (OSError, TypeError, ValueError) as error:
                self.export_error.emit(str(error))
                return

            self.export_finished.emit(os.path.abspath(destination))

        self.web_view.page().printToPdf(pdf_ready)