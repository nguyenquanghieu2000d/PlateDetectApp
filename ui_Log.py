# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Log.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Log(object):
    def __init__(self, text):
        self.text = text

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(399, 266)
        MainWindow.setStyleSheet("background-color: rgb(27, 29, 35); color:white")
        MainWindow.setWindowIcon(QtGui.QIcon("./icon/satisfied.ico"))
        self.window = MainWindow
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.labelBoxBlenderInstalation = QtWidgets.QLabel(self.centralwidget)
        self.labelBoxBlenderInstalation.setGeometry(QtCore.QRect(30, 20, 341, 151))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelBoxBlenderInstalation.setFont(font)
        self.labelBoxBlenderInstalation.setStyleSheet("")
        self.labelBoxBlenderInstalation.setTextFormat(QtCore.Qt.RichText)
        self.labelBoxBlenderInstalation.setAlignment(QtCore.Qt.AlignCenter)
        self.labelBoxBlenderInstalation.setObjectName("labelBoxBlenderInstalation")
        self.btnOK = QtWidgets.QPushButton(self.centralwidget)
        self.btnOK.setGeometry(QtCore.QRect(110, 180, 161, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btnOK.setFont(font)
        self.btnOK.setStyleSheet("QPushButton {\n"
                                 "    border: 2px solid rgb(52, 59, 72);\n"
                                 "    border-radius: 10px;    \n"
                                 "    background-color: rgb(52, 59, 72);\n"
                                 "}\n"
                                 "QPushButton:hover {\n"
                                 "    background-color: rgb(57, 65, 80);\n"
                                 "    border: 2px solid rgb(61, 70, 86);\n"
                                 "}\n"
                                 "QPushButton:pressed {    \n"
                                 "    background-color: rgb(35, 40, 49);\n"
                                 "    border: 2px solid rgb(43, 50, 61);\n"
                                 "}")
        self.btnOK.setObjectName("btnOK")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.event()


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "App Nhận diện khuôn mặt"))
        self.labelBoxBlenderInstalation.setText(
            _translate("MainWindow", "<html><head/><body><p>" + self.text + "</p></body></html>"))
        self.btnOK.setText(_translate("MainWindow", "Xác nhận"))

    def event(self):
        self.btnOK.clicked.connect(self.btnOKOnclick)

    def btnOKOnclick(self):
        self.window.close()



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Log("123")
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())