"""Application entry point for EspectroApp."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication


def close_native_splash() -> None:
    """Close the PyInstaller splash when the main window is visible."""
    try:
        import pyi_splash
    except ImportError:
        return

    try:
        pyi_splash.close()
    except Exception:
        # Startup must continue even if the splash was already closed.
        pass


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("EspectroApp")
    app.setOrganizationName("EspectroApp")

    # Keep heavy imports after QApplication creation so the native
    # PyInstaller splash remains visible during startup.
    from main import MainMenu

    window = MainMenu()
    window.show()
    window.raise_()
    window.activateWindow()

    # Paint the main window before closing the native splash.
    app.processEvents()
    close_native_splash()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
