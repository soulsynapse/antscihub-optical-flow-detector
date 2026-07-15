# main.py
import sys

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(project_dir=".")
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
