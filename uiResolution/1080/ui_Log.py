# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Log.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(639, 425)
        MainWindow.setStyleSheet("background-color: rgb(27, 29, 35); color:white")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.labelBoxBlenderInstalation = QtWidgets.QLabel(self.centralwidget)
        self.labelBoxBlenderInstalation.setGeometry(QtCore.QRect(30, 20, 581, 311))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.labelBoxBlenderInstalation.setFont(font)
        self.labelBoxBlenderInstalation.setStyleSheet("")
        self.labelBoxBlenderInstalation.setTextFormat(QtCore.Qt.RichText)
        self.labelBoxBlenderInstalation.setAlignment(QtCore.Qt.AlignCenter)
        self.labelBoxBlenderInstalation.setObjectName("labelBoxBlenderInstalation")
        self.btnOK = QtWidgets.QPushButton(self.centralwidget)
        self.btnOK.setGeometry(QtCore.QRect(240, 350, 161, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
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

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.labelBoxBlenderInstalation.setText(_translate("MainWindow", "<html><head/><body><p>APP NHẬN DIỆN KHUÔN MẶT</p></body></html>"))
        self.btnOK.setText(_translate("MainWindow", "Xác nhận"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
