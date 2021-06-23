import subprocess

from PyQt5.QtCore import QThread, QObject, pyqtSignal


class appThread(QObject):

    def __init__(self, actionn, pathh):
        super().__init__()
        self.actionn = actionn
        self.path = pathh

    finished = pyqtSignal()
    progress = pyqtSignal(float)

    def run(self):
        """Long-running task."""
        self.path, self.time = self.actionn.detectVideo(self.path, self.progress)

        self.finished.emit()


class LoadModelThread(QObject):
    def __init__(self, actionn):
        super().__init__()
        self.actionn = actionn

    finished = pyqtSignal()

    def run(self):
        """Long-running task."""
        self.actionn()
        self.finished.emit()


class subprocessThread(QObject):
    def __init__(self, path):
        super().__init__()
        self.path = path

    finished = pyqtSignal()

    def run(self):
        subprocess.call(self.path)
        self.finished.emit()
