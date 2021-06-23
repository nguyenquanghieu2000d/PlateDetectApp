from ui_PreStart import Ui_PreStart
from PyQt5 import QtWidgets
if __name__ == "__main__":
    import sys
    loadingScreen = 1
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_PreStart()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())



