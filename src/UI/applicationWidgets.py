import sys
import PyQt6.QtWidgets as wgt
from PyQt6.QtCore import QThread, pyqtSignal
from src.spotipyFunctions import recommendTracks
import io
import contextlib

# Run song recommender on a seperate thread to avoid freezing
class SongRec(QThread):
    update_output = pyqtSignal(str)

    def __init__(self, song_name):
        super().__init__()
        self.song_name = song_name

    def run(self):
        # Create a buffer to retrieve all recommendations from program
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            recommendTracks.runAPIRec(self.song_name)
        output = buffer.getvalue()
        self.update_output.emit(output)

# Button to go to song reccomendation page
class songRecButton():
    def __init__(self, button: wgt.QPushButton, main_window):
        self.button = button
        self.main_window = main_window
        self.button.setText("Go to Spotify Song Recommendation Page")
        self.is_green = False
        self.button.clicked.connect(self.on_click)

    def on_click(self):
        self.main_window.show_song_rec_page()

# Button to go back a page (to main menu)
class backButton():
    def __init__(self, button: wgt.QPushButton):
        self.button = button
        self.button.setText("Go Back a Page")
        self.button.clicked.connect(self.on_click)

    def on_click(self):
        # Go to parent page when clicked
        self.button.parent()

# Button to close the program
class exitButton():
    def __init__(self, button: wgt.QPushButton, main_window):
        self.button = button
        self.main_window = main_window

        button.setText("X")
        button.setFixedSize(40, 40)
        button.setStyleSheet("""
            QPushButton {
                background-color: red;
                color: white;
                border-radius: 20px;
                font-weight: bold;
            }
            QPushButton:hover {background-color: darkred;}""")
        self.button.clicked.connect(self.on_click)
    
    def on_click(self):
        self.main_window.close()

class MainWindow(wgt.QMainWindow):
    def __init__(self):
        super().__init__()

        central_layout = wgt.QVBoxLayout()
        top_layout = wgt.QHBoxLayout()

        self.layout = central_layout

        songRecWgt = wgt.QWidget()
        self.recbutton = songRecButton(wgt.QPushButton(), self)
        central_layout.addWidget(self.recbutton.button)
        songRecWgt.setLayout(central_layout)
        self.setCentralWidget(songRecWgt)

        self.exit_button = exitButton(wgt.QPushButton(), self)
        top_layout.addWidget(self.exit_button.button)

        self.setWindowTitle("MusicAI-Application")
        self.showFullScreen()

    def show_song_rec_page(self):
        # Clear current page content
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

        # Add input box and run button
        self.input_box = wgt.QLineEdit()
        self.input_box.setPlaceholderText("Enter song name")
        self.run_button = wgt.QPushButton("Gimme A Song Rec")
        self.output_box = wgt.QTextEdit()
        self.output_box.setReadOnly(True)

        self.layout.addWidget(self.input_box)
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.output_box)

        self.run_button.clicked.connect(self.run_song_rec)

    def run_song_rec(self):
        song_name = self.input_box.text()
        if not song_name:
            self.output_box.append("Please enter a song name.\n")
            return

        self.output_box.append(f"Running Song Recommendation on: {song_name}\n")
        self.worker = SongRec(song_name)
        self.worker.update_output.connect(self.display_output)
        self.worker.start()

    def display_output(self, text):
        self.output_box.append(text)