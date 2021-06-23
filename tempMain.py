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
        MainWindow.resize(1239, 662)
        MainWindow.setStyleSheet("background-color: rgb(27, 29, 35); color:white")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(0, 0, 1231, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setStyleSheet("")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.frameImage = QtWidgets.QFrame(self.centralwidget)
        self.frameImage.setGeometry(QtCore.QRect(160, 50, 1071, 591))
        self.frameImage.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameImage.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameImage.setObjectName("frameImage")
        self.btnHienThiInput = QtWidgets.QPushButton(self.frameImage)
        self.btnHienThiInput.setGeometry(QtCore.QRect(260, 510, 251, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
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
        self.line.setGeometry(QtCore.QRect(510, 0, 51, 591))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.imgChuaDetect = QtWidgets.QLabel(self.frameImage)
        self.imgChuaDetect.setGeometry(QtCore.QRect(10, 50, 500, 450))
        self.imgChuaDetect.setStyleSheet("QLabel{\n"
"    background: rgb(38, 38, 38)\n"
"}")
        self.imgChuaDetect.setFrameShape(QtWidgets.QFrame.Box)
        self.imgChuaDetect.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imgChuaDetect.setLineWidth(7)
        self.imgChuaDetect.setText("")
        self.imgChuaDetect.setObjectName("imgChuaDetect")
        self.btnDetectImage = QtWidgets.QPushButton(self.frameImage)
        self.btnDetectImage.setGeometry(QtCore.QRect(10, 510, 241, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
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
        self.imgDaDetect.setGeometry(QtCore.QRect(560, 50, 500, 450))
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
        font.setPointSize(14)
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
        self.lbKetQua.setGeometry(QtCore.QRect(566, 2, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lbKetQua.setFont(font)
        self.lbKetQua.setObjectName("lbKetQua")
        self.lbAnhDauVao = QtWidgets.QLabel(self.frameImage)
        self.lbAnhDauVao.setGeometry(QtCore.QRect(190, 0, 321, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lbAnhDauVao.setFont(font)
        self.lbAnhDauVao.setObjectName("lbAnhDauVao")
        self.lbThoiGian = QtWidgets.QLabel(self.frameImage)
        self.lbThoiGian.setGeometry(QtCore.QRect(680, 0, 371, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lbThoiGian.setFont(font)
        self.lbThoiGian.setText("")
        self.lbThoiGian.setObjectName("lbThoiGian")
        self.btnHienThiOutput = QtWidgets.QPushButton(self.frameImage)
        self.btnHienThiOutput.setGeometry(QtCore.QRect(560, 510, 251, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
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
        self.frameVideo.setGeometry(QtCore.QRect(0, 0, 1071, 591))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.frameVideo.setFont(font)
        self.frameVideo.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameVideo.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameVideo.setObjectName("frameVideo")
        self.line_2 = QtWidgets.QFrame(self.frameVideo)
        self.line_2.setGeometry(QtCore.QRect(510, 0, 51, 591))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.videoChuaDetect = QtWidgets.QLabel(self.frameVideo)
        self.videoChuaDetect.setGeometry(QtCore.QRect(10, 50, 500, 450))
        self.videoChuaDetect.setStyleSheet("QLabel{\n"
"    background: rgb(38, 38, 38)\n"
"}")
        self.videoChuaDetect.setFrameShape(QtWidgets.QFrame.Box)
        self.videoChuaDetect.setFrameShadow(QtWidgets.QFrame.Raised)
        self.videoChuaDetect.setLineWidth(7)
        self.videoChuaDetect.setText("")
        self.videoChuaDetect.setObjectName("videoChuaDetect")
        self.btnDetectVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnDetectVideo.setGeometry(QtCore.QRect(10, 560, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
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
        self.videoDaDetect.setGeometry(QtCore.QRect(560, 50, 500, 450))
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
        font.setPointSize(14)
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
        self.lbKetQuaVideo.setGeometry(QtCore.QRect(566, 2, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lbKetQuaVideo.setFont(font)
        self.lbKetQuaVideo.setObjectName("lbKetQuaVideo")
        self.lbVideoInput = QtWidgets.QLabel(self.frameVideo)
        self.lbVideoInput.setGeometry(QtCore.QRect(190, 0, 321, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lbVideoInput.setFont(font)
        self.lbVideoInput.setObjectName("lbVideoInput")
        self.lbThoiGianVideo = QtWidgets.QLabel(self.frameVideo)
        self.lbThoiGianVideo.setGeometry(QtCore.QRect(670, 0, 371, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lbThoiGianVideo.setFont(font)
        self.lbThoiGianVideo.setText("")
        self.lbThoiGianVideo.setObjectName("lbThoiGianVideo")
        self.btnHienThiInputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnHienThiInputVideo.setGeometry(QtCore.QRect(260, 560, 251, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
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
        self.btnHienThiOutputVideo.setGeometry(QtCore.QRect(560, 560, 251, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
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
        self.btnStartInputVideo.setGeometry(QtCore.QRect(160, 510, 61, 41))
        self.btnStartInputVideo.setObjectName("btnStartInputVideo")
        self.btnPauseInputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnPauseInputVideo.setGeometry(QtCore.QRect(230, 510, 61, 41))
        self.btnPauseInputVideo.setObjectName("btnPauseInputVideo")
        self.btnStopInputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnStopInputVideo.setGeometry(QtCore.QRect(300, 510, 61, 41))
        self.btnStopInputVideo.setObjectName("btnStopInputVideo")
        self.btnStopOutputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnStopOutputVideo.setGeometry(QtCore.QRect(850, 510, 61, 41))
        self.btnStopOutputVideo.setObjectName("btnStopOutputVideo")
        self.btnStartOutputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnStartOutputVideo.setGeometry(QtCore.QRect(710, 510, 61, 41))
        self.btnStartOutputVideo.setObjectName("btnStartOutputVideo")
        self.btnPauseOutputVideo = QtWidgets.QPushButton(self.frameVideo)
        self.btnPauseOutputVideo.setGeometry(QtCore.QRect(780, 510, 61, 41))
        self.btnPauseOutputVideo.setObjectName("btnPauseOutputVideo")
        self.pbVideoDetect = QtWidgets.QProgressBar(self.frameVideo)
        self.pbVideoDetect.setGeometry(QtCore.QRect(830, 560, 221, 31))
        self.pbVideoDetect.setProperty("value", 0)
        self.pbVideoDetect.setObjectName("pbVideoDetect")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(0, 50, 151, 591))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.btnLoadModel = QtWidgets.QPushButton(self.frame_2)
        self.btnLoadModel.setGeometry(QtCore.QRect(10, 160, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
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
        self.lbTenModel = QtWidgets.QLabel(self.frame_2)
        self.lbTenModel.setGeometry(QtCore.QRect(10, 110, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lbTenModel.setFont(font)
        self.lbTenModel.setStyleSheet("")
        self.lbTenModel.setObjectName("lbTenModel")
        self.status = QtWidgets.QLabel(self.frame_2)
        self.status.setGeometry(QtCore.QRect(20, 210, 121, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.status.setFont(font)
        self.status.setText("")
        self.status.setObjectName("status")
        self.btnImageMode = QtWidgets.QPushButton(self.frame_2)
        self.btnImageMode.setGeometry(QtCore.QRect(20, 10, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnImageMode.setFont(font)
        self.btnImageMode.setObjectName("btnImageMode")
        self.btnVideoMode = QtWidgets.QPushButton(self.frame_2)
        self.btnVideoMode.setGeometry(QtCore.QRect(20, 60, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnVideoMode.setFont(font)
        self.btnVideoMode.setObjectName("btnVideoMode")
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
        self.btnLoadModel.setText(_translate("MainWindow", "Load model"))
        self.lbTenModel.setText(_translate("MainWindow", "Tên model"))
        self.btnImageMode.setText(_translate("MainWindow", "Nhận diện ảnh"))
        self.btnVideoMode.setText(_translate("MainWindow", "Nhận diện video"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())