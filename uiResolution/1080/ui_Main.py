# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Main.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1570, 884)
        MainWindow.setStyleSheet("background-color: rgb(27, 29, 35); color:white")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(0, 0, 1561, 81))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setStyleSheet("")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.frameImage = QtWidgets.QFrame(self.centralwidget)
        self.frameImage.setGeometry(QtCore.QRect(270, 90, 1291, 771))
        self.frameImage.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameImage.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameImage.setObjectName("frameImage")
        self.btnHienThiInput = QtWidgets.QPushButton(self.frameImage)
        self.btnHienThiInput.setGeometry(QtCore.QRect(310, 690, 301, 81))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.btnHienThiInput.setFont(font)
        self.btnHienThiInput.setStyleSheet("QPushButton {\n"
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
        self.btnHienThiInput.setObjectName("btnHienThiInput")
        self.line = QtWidgets.QFrame(self.frameImage)
        self.line.setGeometry(QtCore.QRect(610, 0, 51, 781))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.imgChuaDetect = QtWidgets.QLabel(self.frameImage)
        self.imgChuaDetect.setGeometry(QtCore.QRect(10, 50, 600, 600))
        self.imgChuaDetect.setStyleSheet("QLabel{\n"
                                         "    background: rgb(38, 38, 38)\n"
                                         "}")
        self.imgChuaDetect.setFrameShape(QtWidgets.QFrame.Box)
        self.imgChuaDetect.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imgChuaDetect.setLineWidth(7)
        self.imgChuaDetect.setText("")
        self.imgChuaDetect.setObjectName("imgChuaDetect")
        self.btnDetectImage = QtWidgets.QPushButton(self.frameImage)
        self.btnDetectImage.setGeometry(QtCore.QRect(10, 690, 291, 81))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.btnDetectImage.setFont(font)
        self.btnDetectImage.setStyleSheet("QPushButton {\n"
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
        self.btnDetectImage.setObjectName("btnDetectImage")
        self.imgDaDetect = QtWidgets.QLabel(self.frameImage)
        self.imgDaDetect.setGeometry(QtCore.QRect(660, 50, 600, 600))
        self.imgDaDetect.setStyleSheet("QLabel{\n"
                                       "    background: rgb(38, 38, 38)\n"
                                       "}")
        self.imgDaDetect.setFrameShape(QtWidgets.QFrame.Box)
        self.imgDaDetect.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imgDaDetect.setLineWidth(7)
        self.imgDaDetect.setText("")
        self.imgDaDetect.setObjectName("imgDaDetect")
        self.btnTaiAnh = QtWidgets.QPushButton(self.frameImage)
        self.btnTaiAnh.setGeometry(QtCore.QRect(10, 0, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.btnTaiAnh.setFont(font)
        self.btnTaiAnh.setStyleSheet("QPushButton {\n"
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
        self.btnTaiAnh.setObjectName("btnTaiAnh")
        self.lbKetQua = QtWidgets.QLabel(self.frameImage)
        self.lbKetQua.setGeometry(QtCore.QRect(660, 0, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lbKetQua.setFont(font)
        self.lbKetQua.setObjectName("lbKetQua")
        self.lbAnhDauVao = QtWidgets.QLabel(self.frameImage)
        self.lbAnhDauVao.setGeometry(QtCore.QRect(190, 0, 321, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lbAnhDauVao.setFont(font)
        self.lbAnhDauVao.setObjectName("lbAnhDauVao")
        self.lbThoiGian = QtWidgets.QLabel(self.frameImage)
        self.lbThoiGian.setGeometry(QtCore.QRect(810, 0, 371, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lbThoiGian.setFont(font)
        self.lbThoiGian.setText("")
        self.lbThoiGian.setObjectName("lbThoiGian")
        self.btnHienThiOutput = QtWidgets.QPushButton(self.frameImage)
        self.btnHienThiOutput.setGeometry(QtCore.QRect(660, 690, 371, 81))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.btnHienThiOutput.setFont(font)
        self.btnHienThiOutput.setStyleSheet("QPushButton {\n"
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
        self.btnHienThiOutput.setObjectName("btnHienThiOutput")
        self.frameVideo = QtWidgets.QFrame(self.frameImage)
        self.frameVideo.setGeometry(QtCore.QRect(0, 0, 1291, 791))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.frameVideo.setFont(font)
        self.frameVideo.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameVideo.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameVideo.setObjectName("frameVideo")
        self.line_2 = QtWidgets.QFrame(self.frameVideo)
        self.line_2.setGeometry(QtCore.QRect(610, 0, 51, 791))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.videoChuaDetect = QtWidgets.QLabel(self.frameVideo)
        self.videoChuaDetect.setGeometry(QtCore.QRect(10, 50, 600, 600))
        self.videoChuaDetect.setStyleSheet("QLabel{\n"
                                           "    background: rgb(38, 38, 38)\n"
                                           "}")
        self.videoChuaDetect.setFrameShape(QtWidgets.QFrame.Box)
        self.videoChuaDetect.setFrameShadow(QtWidgets.QFrame.Raised)
        self.videoChuaDetect.setLineWidth(7)
        self.videoChuaDetect.setText("")
        self.videoChuaDetect.setObjectName("videoChuaDetect")
        self.btnDetectVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnDetectVideo.setGeometry(QtCore.QRect(10, 720, 291, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.btnDetectVideo.setFont(font)
        self.btnDetectVideo.setStyleSheet("QPushButton {\n"
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
        self.btnDetectVideo.setObjectName("btnDetectVideo")
        self.videoDaDetect = QtWidgets.QLabel(self.frameVideo)
        self.videoDaDetect.setGeometry(QtCore.QRect(660, 50, 600, 600))
        self.videoDaDetect.setStyleSheet("QLabel{\n"
                                         "    background: rgb(38, 38, 38)\n"
                                         "}")
        self.videoDaDetect.setFrameShape(QtWidgets.QFrame.Box)
        self.videoDaDetect.setFrameShadow(QtWidgets.QFrame.Raised)
        self.videoDaDetect.setLineWidth(7)
        self.videoDaDetect.setText("")
        self.videoDaDetect.setObjectName("videoDaDetect")
        self.btnTaiVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnTaiVideo.setGeometry(QtCore.QRect(10, 0, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.btnTaiVideo.setFont(font)
        self.btnTaiVideo.setStyleSheet("QPushButton {\n"
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
        self.btnTaiVideo.setObjectName("btnTaiVideo")
        self.lbKetQuaVideo = QtWidgets.QLabel(self.frameVideo)
        self.lbKetQuaVideo.setGeometry(QtCore.QRect(660, 0, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lbKetQuaVideo.setFont(font)
        self.lbKetQuaVideo.setObjectName("lbKetQuaVideo")
        self.lbVideoInput = QtWidgets.QLabel(self.frameVideo)
        self.lbVideoInput.setGeometry(QtCore.QRect(190, 0, 321, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lbVideoInput.setFont(font)
        self.lbVideoInput.setObjectName("lbVideoInput")
        self.lbThoiGianVideo = QtWidgets.QLabel(self.frameVideo)
        self.lbThoiGianVideo.setGeometry(QtCore.QRect(764, -2, 371, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lbThoiGianVideo.setFont(font)
        self.lbThoiGianVideo.setText("")
        self.lbThoiGianVideo.setObjectName("lbThoiGianVideo")
        self.btnHienThiInputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnHienThiInputVideo.setGeometry(QtCore.QRect(310, 720, 301, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.btnHienThiInputVideo.setFont(font)
        self.btnHienThiInputVideo.setStyleSheet("QPushButton {\n"
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
        self.btnHienThiInputVideo.setObjectName("btnHienThiInputVideo")
        self.btnHienThiOutputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnHienThiOutputVideo.setGeometry(QtCore.QRect(660, 720, 301, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.btnHienThiOutputVideo.setFont(font)
        self.btnHienThiOutputVideo.setStyleSheet("QPushButton {\n"
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
        self.btnHienThiOutputVideo.setObjectName("btnHienThiOutputVideo")
        self.btnStartInputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnStartInputVideo.setGeometry(QtCore.QRect(190, 660, 61, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnStartInputVideo.setFont(font)
        self.btnStartInputVideo.setObjectName("btnStartInputVideo")
        self.btnPauseInputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnPauseInputVideo.setGeometry(QtCore.QRect(260, 660, 61, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnPauseInputVideo.setFont(font)
        self.btnPauseInputVideo.setObjectName("btnPauseInputVideo")
        self.btnStopInputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnStopInputVideo.setGeometry(QtCore.QRect(330, 660, 61, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnStopInputVideo.setFont(font)
        self.btnStopInputVideo.setObjectName("btnStopInputVideo")
        self.btnStopOutputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnStopOutputVideo.setGeometry(QtCore.QRect(1000, 670, 61, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnStopOutputVideo.setFont(font)
        self.btnStopOutputVideo.setObjectName("btnStopOutputVideo")
        self.btnStartOutputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnStartOutputVideo.setGeometry(QtCore.QRect(860, 670, 61, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnStartOutputVideo.setFont(font)
        self.btnStartOutputVideo.setObjectName("btnStartOutputVideo")
        self.btnPauseOutputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnPauseOutputVideo.setGeometry(QtCore.QRect(930, 670, 61, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnPauseOutputVideo.setFont(font)
        self.btnPauseOutputVideo.setObjectName("btnPauseOutputVideo")
        self.pbVideoDetect = QtWidgets.QProgressBar(self.frameVideo)
        self.pbVideoDetect.setGeometry(QtCore.QRect(980, 720, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pbVideoDetect.setFont(font)
        self.pbVideoDetect.setProperty("value", 0)
        self.pbVideoDetect.setObjectName("pbVideoDetect")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(0, 90, 261, 771))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.lbTenModel = QtWidgets.QLabel(self.frame_2)
        self.lbTenModel.setGeometry(QtCore.QRect(30, 200, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.lbTenModel.setFont(font)
        self.lbTenModel.setStyleSheet("")
        self.lbTenModel.setAlignment(QtCore.Qt.AlignCenter)
        self.lbTenModel.setObjectName("lbTenModel")
        self.status = QtWidgets.QLabel(self.frame_2)
        self.status.setGeometry(QtCore.QRect(70, 390, 121, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.status.setFont(font)
        self.status.setText("")
        self.status.setObjectName("status")
        self.btnVideoMode = QtWidgets.QPushButton(self.frame_2)
        self.btnVideoMode.setGeometry(QtCore.QRect(30, 110, 201, 71))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.btnVideoMode.setFont(font)
        self.btnVideoMode.setObjectName("btnVideoMode")
        self.btnImageMode = QtWidgets.QPushButton(self.frame_2)
        self.btnImageMode.setGeometry(QtCore.QRect(30, 30, 201, 71))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.btnImageMode.setFont(font)
        self.btnImageMode.setObjectName("btnImageMode")
        self.btnLoadModel = QtWidgets.QPushButton(self.frame_2)
        self.btnLoadModel.setGeometry(QtCore.QRect(30, 250, 201, 71))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.btnLoadModel.setFont(font)
        self.btnLoadModel.setStyleSheet("QPushButton {\n"
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
        self.btnLoadModel.setObjectName("btnLoadModel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.title.setText(_translate("MainWindow", "APP ĐỊNH VỊ BIỂN SỐ XE"))
        self.btnHienThiInput.setText(_translate("MainWindow", "Hiển thị"))
        self.btnDetectImage.setText(_translate("MainWindow", "Nhận diện"))
        self.btnTaiAnh.setText(_translate("MainWindow", "Tải ảnh nào"))
        self.lbKetQua.setText(_translate("MainWindow", "Kết quả"))
        self.lbAnhDauVao.setText(_translate("MainWindow", "Ảnh đầu vào"))
        self.btnHienThiOutput.setText(_translate("MainWindow", "Hiển thị"))
        self.btnDetectVideo.setText(_translate("MainWindow", "Nhận diện"))
        self.btnTaiVideo.setText(_translate("MainWindow", "Tải video nào"))
        self.lbKetQuaVideo.setText(_translate("MainWindow", "Kết quả"))
        self.lbVideoInput.setText(_translate("MainWindow", "Video đầu vào"))
        self.btnHienThiInputVideo.setText(_translate("MainWindow", "Hiển thị"))
        self.btnHienThiOutputVideo.setText(_translate("MainWindow", "Hiển thị"))
        self.btnStartInputVideo.setText(_translate("MainWindow", "Start"))
        self.btnPauseInputVideo.setText(_translate("MainWindow", "Pause"))
        self.btnStopInputVideo.setText(_translate("MainWindow", "Stop"))
        self.btnStopOutputVideo.setText(_translate("MainWindow", "Stop"))
        self.btnStartOutputVideo.setText(_translate("MainWindow", "Start"))
        self.btnPauseOutputVideo.setText(_translate("MainWindow", "Pause"))
        self.lbTenModel.setText(_translate("MainWindow", "Tên model"))
        self.btnVideoMode.setText(_translate("MainWindow", "Nhận diện video"))
        self.btnImageMode.setText(_translate("MainWindow", "Nhận diện ảnh"))
        self.btnLoadModel.setText(_translate("MainWindow", "Load model"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())