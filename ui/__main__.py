"""Launch the application: ``python -m ui [settings.json]``."""

import sys

from PySide6.QtWidgets import QApplication

from config.settings import load_settings
from ui.main_window import APP_TITLE, MainWindow
from ui.theme import apply_dark_theme


def main(argv: list[str]) -> int:
    settings = load_settings(argv[1] if len(argv) > 1 else None)
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(APP_TITLE)
    apply_dark_theme(app)

    window = MainWindow(settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
