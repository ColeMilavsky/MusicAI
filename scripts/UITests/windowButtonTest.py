import PyQt6.QtWidgets as wgt

class MyButton():
    def __init__(self, button: wgt.QPushButton):
        self.button = button
        self.button.setText("Click Me!")
        self.is_green = False
        self.button.clicked.connect(self.on_click)

    def on_click(self):
        if self.is_green:
            self.button.setStyleSheet("background-color: red; color: white;")
            self.button.setText("Now Red!")
        else:
            self.button.setStyleSheet("background-color: green; color: white;")
            self.button.setText("Now Green!")
        self.is_green = not self.is_green

class MainWindow(wgt.QMainWindow):
    def __init__(self):
        super().__init__()

        central_widget = wgt.QWidget()
        layout = wgt.QVBoxLayout()

        self.mybutton = MyButton(wgt.QPushButton())

        layout.addWidget(self.mybutton.button)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.setWindowTitle("Main-Menu")
        self.resize(300, 200)
        self.show()