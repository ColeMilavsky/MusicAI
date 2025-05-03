import sys
import PyQt6.QtWidgets as wgt
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from src.spotipyFunctions import recommendTracks
import io
import contextlib

# Run song recommender on a seperate thread to avoid freezing
class SongRec(QThread):
    update_output = pyqtSignal(str)

    def __init__(self, song_name, sim_score):
        super().__init__()
        self.song_name = song_name
        self.sim_score = sim_score

    def run(self):
        # Create a buffer to retrieve all recommendations from program
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            recommendTracks.runAPIRec(self.song_name, self.sim_score)
        output = buffer.getvalue()
        self.update_output.emit(output)

# Button to go to song reccomendation page
class songRecButton():
    def __init__(self, button: wgt.QPushButton, main_window):
        self.button = button
        self.main_window = main_window
        self.button.setText("Enter a song to get recs")
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
                border-radius: 0px;
                font-weight: bold;
            }
            QPushButton:hover {background-color: darkred;}""")
        self.button.clicked.connect(self.on_click)
    
    def on_click(self):
        self.main_window.close()

class MainWindow(wgt.QMainWindow):
    def __init__(self):
        super().__init__()

        # Setup central widget
        self.central_widget = wgt.QWidget()
        self.layout = wgt.QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.setWindowTitle("MusicAI-Application")
        self.showFullScreen()

        # Setup menu bar
        self.setup_menu()

        self.label = wgt.QLabel("Welcome to MusicAI!", alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)

    def setup_menu(self):
        # Create main menu bar
        menubar = self.menuBar()

        # Add "File" menu
        file_menu = menubar.addMenu("File")

        # Add "Exit" action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Add "File" menu
        app_menu = menubar.addMenu("Apps")

        # Add "Song Recommendations" action
        rec_action = QAction("Song Recommendations", self)
        rec_action.triggered.connect(self.show_song_rec_page)
        app_menu.addAction(rec_action)

        # Add "Help" menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_page)
        help_menu.addAction(about_action)

    def show_about_page(self):
        wgt.QMessageBox.information(self, "About", "This is the MusicAI application.\nVersion 1.0")

    def show_song_rec_page(self):
        # Clear current page content
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

        # Add input box and run button
        self.input_box1 = wgt.QLineEdit()
        self.input_box2 = wgt.QLineEdit()
        self.input_box1.setPlaceholderText("Enter song name")
        self.input_box2.setPlaceholderText("Enter similarity score(0-5)")
        self.run_button = wgt.QPushButton("Gimme A Song Rec")
        self.output_box = wgt.QTextBrowser()
        self.output_box.setReadOnly(True)
        self.output_box.setOpenExternalLinks(True)

        self.layout.addWidget(self.input_box1)
        self.layout.addWidget(self.input_box2)
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.output_box)

        self.run_button.clicked.connect(self.run_song_rec)

    def run_song_rec(self):
        song_name = self.input_box1.text()
        if not song_name:
            self.output_box.append("Please enter a song name.\n")
            return
        sim_score = self.input_box2.text()
        if not sim_score.isdigit() or int(sim_score) not in range(0, 6):
            self.output_box.append("Please enter a similarity score 0 to 5.\n")
            return

        self.output_box.append(f"Searching for {song_name}...\n")
        self.worker = SongRec(song_name, int(sim_score))
        self.worker.update_output.connect(self.display_output)
        self.worker.start()

    def display_output(self, text):
        for line in text.splitlines():
            self.output_box.append(line)