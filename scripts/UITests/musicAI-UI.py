import PyQt6.QtWidgets as wgt
import sys
import scripts.UITests.windowButtonTest as wb

app = wgt.QApplication(sys.argv)
app.setStyle("Fusion")

window = wb.MainWindow()

app.exec()