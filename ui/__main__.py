"""Launch the image selection dialog: ``python -m ui [settings.json]``."""

import sys

from PySide6.QtWidgets import QApplication, QDialog

from config.settings import load_settings
from ui.image_selection_dialog import ImageSelectionDialog


def main(argv: list[str]) -> int:
    settings = load_settings(argv[1] if len(argv) > 1 else None)
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(settings.app_name)

    dialog = ImageSelectionDialog(settings)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        print("Selección cancelada.")
        return 1

    before, after = dialog.selected_pair()
    for label, acquired in (("Antes", before), ("Después", after)):
        image = acquired.image
        warnings = f" [ADVERTENCIA] {'; '.join(acquired.warnings)}" if acquired.warnings else ""
        print(f"{label}: {image.source} ({image.width}x{image.height}){warnings}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
