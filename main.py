import sys
from PyQt6.QtWidgets import QApplication
from src.gui import LiveDocApp


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("LiveDoc")
    app.setApplicationDisplayName("LiveDoc — Intelligent Document Digitizer")

    window = LiveDocApp()
    window.show()

    sys.exit(app.exec())
