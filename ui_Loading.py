from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QWidget, QLabel


class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)
        self.label_animation = QLabel(self)
        self.movie = QMovie("./icon/Loading_3.gif")
        self.label_animation.setMovie(self.movie)

    def startAnimation(self):
        self.show()


    def stopAnimation(self):
        self.movie.stop()
        self.close()
