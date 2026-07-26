from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class SplashScreen(QWidget):
    """Centered startup splash screen for EspectroApp."""

    def __init__(
        self,
        logo_path: str | Path,
        version: str = "1.0.0",
    ) -> None:
        super().__init__()

        self.logo_path = Path(logo_path).resolve()
        self.version = str(version)

        self._configure_window()
        self._build_interface()

    def _configure_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def _build_interface(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.container = QFrame(self)
        self.container.setObjectName("splashContainer")
        self.container.setFixedSize(820, 470)
        self.container.setStyleSheet(
            """
            QFrame#splashContainer {
                background: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 1, y2: 1,
                    stop: 0 #F8FBFA,
                    stop: 0.55 #EEF6F3,
                    stop: 1 #DCEEE8
                );
                border: 2px solid #0F7F69;
                border-radius: 24px;
            }

            QLabel {
                background: transparent;
            }

            QProgressBar {
                min-height: 14px;
                max-height: 14px;
                border: none;
                border-radius: 7px;
                background-color: #C7DDD6;
                color: transparent;
            }

            QProgressBar::chunk {
                border-radius: 7px;
                background: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 1, y2: 0,
                    stop: 0 #0F8A6B,
                    stop: 1 #42D3AE
                );
            }
            """
        )

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(34, 22, 34, 22)
        layout.setSpacing(7)

        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setFixedHeight(285)

        pixmap = QPixmap(str(self.logo_path))

        if pixmap.isNull():
            self.logo_label.setText("EspectroApp")
            self.logo_label.setStyleSheet(
                "color: #073F36; font-size: 48px; font-weight: 700;"
            )
            print(f"[Splash] Logo could not be loaded: {self.logo_path}")
        else:
            pixmap = pixmap.scaled(
                740,
                270,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.logo_label.setPixmap(pixmap)

        subtitle = QLabel("Spectral Analysis Suite")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(
            "font-size: 14px; font-weight: 500; color: #42675D;"
        )

        self.status_label = QLabel("Starting EspectroApp...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 14px; font-weight: 600; color: #163B32;"
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)

        self.percentage_label = QLabel("0%")
        self.percentage_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.percentage_label.setStyleSheet(
            "font-size: 11px; font-weight: 600; color: #42675D;"
        )

        version_label = QLabel(f"Version {self.version}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet(
            "font-size: 10px; color: #6C8B83;"
        )

        layout.addWidget(self.logo_label)
        layout.addWidget(subtitle)
        layout.addSpacing(8)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.percentage_label)
        layout.addWidget(version_label)

        root_layout.addWidget(self.container)

    def show_centered(self) -> None:
        """Show the splash overlay with the loading card centered."""
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None

        if screen is not None:
            self.setGeometry(screen.geometry())
        else:
            self.showFullScreen()
            return

        self.show()
        self.raise_()
        self.activateWindow()

        QTimer.singleShot(0, self._restore_screen_geometry)
        QTimer.singleShot(50, self._restore_screen_geometry)

    def _restore_screen_geometry(self) -> None:
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is not None:
            self.setGeometry(screen.geometry())

    def update_progress(self, value: int, message: str) -> None:
        safe_value = max(0, min(100, int(value)))
        self.progress_bar.setValue(safe_value)
        self.percentage_label.setText(f"{safe_value}%")
        self.status_label.setText(str(message))
