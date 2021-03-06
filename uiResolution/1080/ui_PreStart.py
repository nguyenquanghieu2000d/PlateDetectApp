# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PreStart.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1798, 899)
        MainWindow.setStyleSheet("background-color: rgb(27, 29, 35); color:white")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnBatDau = QtWidgets.QPushButton(self.centralwidget)
        self.btnBatDau.setGeometry(QtCore.QRect(520, 740, 731, 91))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.btnBatDau.setFont(font)
        self.btnBatDau.setStyleSheet("QPushButton {\n"
"    border: 2px solid rgb(52, 59, 72);\n"
"    border-radius: 5px;    \n"
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
        self.btnBatDau.setObjectName("btnBatDau")
        self.labelBoxBlenderInstalation = QtWidgets.QLabel(self.centralwidget)
        self.labelBoxBlenderInstalation.setGeometry(QtCore.QRect(330, 50, 1131, 91))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.labelBoxBlenderInstalation.setFont(font)
        self.labelBoxBlenderInstalation.setStyleSheet("")
        self.labelBoxBlenderInstalation.setAlignment(QtCore.Qt.AlignCenter)
        self.labelBoxBlenderInstalation.setObjectName("labelBoxBlenderInstalation")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(520, 200, 731, 511))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.imgImageMode_2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(self.imgImageMode_2.sizePolicy().hasHeightForWidth())
        self.imgImageMode_2.setSizePolicy(sizePolicy)
        self.imgImageMode_2.setFrameShape(QtWidgets.QFrame.Box)
        self.imgImageMode_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imgImageMode_2.setLineWidth(7)
        self.imgImageMode_2.setText("")
        self.imgImageMode_2.setObjectName("imgImageMode_2")
        self.verticalLayout_2.addWidget(self.imgImageMode_2)
        self.labelBoxBlenderInstalation_7 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(16)
        sizePolicy.setHeightForWidth(self.labelBoxBlenderInstalation_7.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_7.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.labelBoxBlenderInstalation_7.setFont(font)
        self.labelBoxBlenderInstalation_7.setStyleSheet("")
        self.labelBoxBlenderInstalation_7.setAlignment(QtCore.Qt.AlignCenter)
        self.labelBoxBlenderInstalation_7.setObjectName("labelBoxBlenderInstalation_7")
        self.verticalLayout_2.addWidget(self.labelBoxBlenderInstalation_7)
        self.labelBoxBlenderInstalation_8 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(16)
        sizePolicy.setHeightForWidth(self.labelBoxBlenderInstalation_8.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_8.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.labelBoxBlenderInstalation_8.setFont(font)
        self.labelBoxBlenderInstalation_8.setStyleSheet("")
        self.labelBoxBlenderInstalation_8.setAlignment(QtCore.Qt.AlignCenter)
        self.labelBoxBlenderInstalation_8.setObjectName("labelBoxBlenderInstalation_8")
        self.verticalLayout_2.addWidget(self.labelBoxBlenderInstalation_8)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.verticalLayout.setObjectName("verticalLayout")
        self.imgImageMode = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(self.imgImageMode.sizePolicy().hasHeightForWidth())
        self.imgImageMode.setSizePolicy(sizePolicy)
        self.imgImageMode.setFrameShape(QtWidgets.QFrame.Box)
        self.imgImageMode.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imgImageMode.setLineWidth(7)
        self.imgImageMode.setText("")
        self.imgImageMode.setObjectName("imgImageMode")
        self.verticalLayout.addWidget(self.imgImageMode)
        self.labelBoxBlenderInstalation_5 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(16)
        sizePolicy.setHeightForWidth(self.labelBoxBlenderInstalation_5.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_5.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.labelBoxBlenderInstalation_5.setFont(font)
        self.labelBoxBlenderInstalation_5.setStyleSheet("")
        self.labelBoxBlenderInstalation_5.setAlignment(QtCore.Qt.AlignCenter)
        self.labelBoxBlenderInstalation_5.setObjectName("labelBoxBlenderInstalation_5")
        self.verticalLayout.addWidget(self.labelBoxBlenderInstalation_5)
        self.labelBoxBlenderInstalation_6 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(16)
        sizePolicy.setHeightForWidth(self.labelBoxBlenderInstalation_6.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_6.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.labelBoxBlenderInstalation_6.setFont(font)
        self.labelBoxBlenderInstalation_6.setStyleSheet("")
        self.labelBoxBlenderInstalation_6.setAlignment(QtCore.Qt.AlignCenter)
        self.labelBoxBlenderInstalation_6.setObjectName("labelBoxBlenderInstalation_6")
        self.verticalLayout.addWidget(self.labelBoxBlenderInstalation_6)
        self.horizontalLayout.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnBatDau.setText(_translate("MainWindow", "B???T ?????U"))
        self.labelBoxBlenderInstalation.setText(_translate("MainWindow", "APP ?????NH V??? BI???N S??? XE"))
        self.labelBoxBlenderInstalation_7.setText(_translate("MainWindow", "NGUY???N QUANG HI???U"))
        self.labelBoxBlenderInstalation_8.setText(_translate("MainWindow", "D13CNPM7"))
        self.labelBoxBlenderInstalation_5.setText(_translate("MainWindow", "NG?? TH??? HU???"))
        self.labelBoxBlenderInstalation_6.setText(_translate("MainWindow", "D13CNPM5"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
