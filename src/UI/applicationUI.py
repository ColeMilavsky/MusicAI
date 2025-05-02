import PyQt6.QtWidgets as wgt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.UI import applicationWidgets as aw

app = wgt.QApplication(sys.argv)

window = aw.MainWindow()

app.exec()