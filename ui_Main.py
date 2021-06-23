# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Main - Copy.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import os
import sys
from time import sleep

from dotenv import load_dotenv

load_dotenv()
WMPATH = os.getenv("WMPATH")
YOLOV5_PATH = os.getenv("YOLOV5_PATH")
sys.path.insert(0, YOLOV5_PATH)
import cv2
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QVBoxLayout

from appThread import appThread, LoadModelThread, subprocessThread
from chooseFile import Dialogg
from ui_Log import Ui_Log
from yolov5 import detect


class Ui_Main(object):
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

        # User modify
        self.model = ""
        self.modelPath = ""
        self.ImagePath = ""
        self.DetectImage = None
        self.VideoPath = None
        self.setupVideo()
        self.event()
        self.btnTaiVideoEnable = True
        self.btnDetectVideoEnable = True
        self.btnHienThiOutputVideo = True
        self.btnLoadModelEnable = True

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

    def LoadModelAction(self):
        self.lbTenModel.setText("Đang tải model ...")
        self.btnDetectImage.setEnabled(False)
        self.model = detect.htldYolov5Detect()
        self.model.LoadModel(self.modelPath)
        self.lbTenModel.setText("Đang dùng: " + self.modelPath.split("/")[-1])
        self.btnDetectImage.setEnabled(True)


    def SetEnableLoadModel(self):
        self.btnLoadModelEnable = True
        self.thongBao("Load Model thành công")

    def LoadModel(self):
        if self.btnLoadModelEnable == False:
            self.thongBao("Đang loadmodel xin chờ")
            return

        try:
            self.ui = Dialogg()
            self.modelPath = self.ui.openFileNameDialog()
        except:
            self.modelPath = ""

        try:
            if self.modelPath:
                self.worker1 = LoadModelThread(self.LoadModelAction)
                self.thread1 = QThread()
                self.worker1.moveToThread(self.thread1)
                self.thread1.started.connect(self.worker1.run)
                self.worker1.finished.connect(self.thread1.quit)
                self.worker1.finished.connect(self.worker1.deleteLater)
                self.thread1.finished.connect(self.thread1.deleteLater)
                self.thread1.start()

                self.btnLoadModelEnable = False
                self.thread1.finished.connect(self.SetEnableLoadModel)
        except:
            string = "Truyền sai định dạng hãy sử dụng model Yolov5 !!"
            self.thongBao(string)
            self.lbTenModel.setText(string)
            self.btnLoadModelEnable = True

    def thongBao(self, text):
        win = QtWidgets.QMainWindow()
        self.ui = Ui_Log(text)
        self.ui.setupUi(win)
        win.show()

    def predictPlate(self):
        img, time = self.model.detectImage(self.ImagePath)
        return img, time

    def display(self, img, image):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA888
            else:
                qformat = QImage.Format_RGB888
        img = cv2.resize(img, (400, 400))

        imag = QImage(img, img.shape[1], img.shape[0], qformat)
        imag = imag.rgbSwapped()

        image.setPixmap(QPixmap.fromImage(imag))

    def TaiAnhAction(self):
        try:
            if self.ImagePath:
                self.ImageModeMainImage = cv2.imread(self.ImagePath)
                self.display(self.ImageModeMainImage, self.imgChuaDetect)
                self.btnDetectImage.setEnabled(True)
        except:
            string = "Sai định dạng hãy chọn ảnh nhé !!"
            self.ImagePath = ""
            self.thongBao(string)

    def btnTaiAnhOnclick(self):
        try:
            self.ui = Dialogg()
            self.ImagePath = self.ui.openFileNameDialog()
        except:
            self.ImagePath = ""
        self.TaiAnhAction()

    def NhanDienImage(self):
        if self.modelPath == "" or self.modelPath is None:
            string = "Hãy load model vào nhé"
            self.thongBao(string)
            return

        if self.ImagePath:
            frame, time = self.predictPlate()
            self.DetectImage = frame
            self.lbThoiGian.setText("Thời gian định vị: " + str(time) + " giây")
            self.display(frame, self.imgDaDetect)
        else:
            string = "Hãy tải ảnh lên"
            self.thongBao(string)
        pass

    def btnHienThiInputOnClick(self):
        if self.ImagePath:
            image = cv2.imread(self.ImagePath)
            plt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(plt_image)
            im_pil.show()
            pass
        else:
            self.thongBao("Hãy tải ảnh lên")
            pass

    def btnHienThiOutputOnClick(self):
        if self.DetectImage is not None:
            plt_image = cv2.cvtColor(self.DetectImage, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(plt_image)
            im_pil.show()
            pass
        else:
            self.thongBao("Bạn chưa detect ra ảnh")
            pass

    def DetectVideoMode(self):
        if self.VideoPath is not None:
            self.TaiVideoAction()
        self.frameVideo.show()
        # self.frameImage.hide()

    def DetectImageMode(self):
        if self.ImagePath is not None:
            self.TaiAnhAction()
        self.frameImage.show()
        self.frameVideo.hide()

    def setupVideo(self):
        self.videoInput = self.makeVideoWidget(self.videoChuaDetect)
        self.mediaPlayerInput = self.makeMediaPlayer(self.videoInput)
        self.videoOutput = self.makeVideoWidget(self.videoDaDetect)
        self.mediaPlayerOutput = self.makeMediaPlayer(self.videoOutput)

    def makeMediaPlayer(self, video):
        mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        mediaPlayer.setVideoOutput(video)
        return mediaPlayer

    def makeVideoWidget(self, video):
        videoOutput = QVideoWidget()
        vbox = QVBoxLayout()
        vbox.addWidget(videoOutput)
        video.setLayout(vbox)
        return videoOutput

    def TaiVideoAction(self):
        if self.VideoPath == "" or self.VideoPath is None:
            return
        try:
            types = ["mp4"]
            for i in types:
                assert self.VideoPath.split(".")[-1] == i
            self.mediaPlayerInput.setMedia(QMediaContent(QUrl(self.VideoPath)))
            self.mediaPlayerInput.play()
        except:
            string = "Sai định dạng hãy chọn video nhé !!"
            self.VideoPath = ""
            self.thongBao(string)

    def TaiVideo(self):
        if self.btnTaiVideoEnable == False:
            string = "Video trước đang được xử lý xin hãy đợi"
            self.thongBao(string)
            return
        try:
            self.ui = Dialogg()
            string = self.ui.openFileNameDialog()
            if string == "" or string is None:
                return
            self.VideoPath = string
        except:
            pass
        self.TaiVideoAction()

    def reportProgress(self, n):
        self.pbVideoDetect.setValue(n)


    def loadDetectVideoAction(self):
        self.mediaPlayerOutput.setMedia(QMediaContent(QUrl(self.detectPath)))
        self.mediaPlayerOutput.play()

    def assignValue(self):
        self.detectPath = self.worker.path
        self.detectPath = "/".join(self.detectPath.split("\\"))
        self.lbThoiGianVideo.setText("Thời gian định vị: " + str(self.worker.time) + " giây")
        print(self.detectPath)
        self.loadDetectVideoAction()


    def SetEnableNhanDienVideo(self):
        self.btnDetectVideoEnable = True
        self.btnTaiVideoEnable = True
        self.thongBao("Định vị thành công")

    def NhanDienVideo(self):
        if self.btnDetectVideoEnable == False:
            string = "Video trước đang được xử lý xin hãy đợi"
            self.thongBao(string)
            return

        if self.modelPath == "" or self.modelPath is None:
            string = "Hãy load model vào nhé"
            self.thongBao(string)
            return

        if self.VideoPath:
            self.lbThoiGianVideo.setText("Đang nhận diện xin chờ")
            self.worker = appThread(self.model, self.VideoPath)
            self.thread = QThread()
            self.worker.moveToThread(self.thread)
            self.thread.start()
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.reportProgress)
            self.thread.finished.connect(self.assignValue)
            self.thread.start()

            # status
            self.btnDetectVideoEnable = False
            self.thread.finished.connect(self.SetEnableNhanDienVideo)
            self.btnTaiVideoEnable = False


        else:
            string = "Hãy tải video lên"
            self.thongBao(string)

    def btnHienThiInputVideoOnclick(self):
        if self.VideoPath:
            self.worker2 = subprocessThread(WMPATH + ' /play /close ' + self.VideoPath)
            self.thread2 = QThread()
            self.worker2.moveToThread(self.thread2)
            self.thread2.started.connect(self.worker2.run)
            self.worker2.finished.connect(self.thread2.quit)
            self.worker2.finished.connect(self.worker2.deleteLater)
            self.thread2.finished.connect(self.thread2.deleteLater)
            self.thread2.start()
        else:
            self.thongBao("Hãy tải video lên")

    def btnHienThiOutputVideoOnclick(self):
        if self.detectPath:
            self.worker3 = subprocessThread(WMPATH + ' /play /close ' + self.detectPath)
            self.thread3 = QThread()
            self.worker3.moveToThread(self.thread3)
            self.thread3.started.connect(self.worker3.run)
            self.worker3.finished.connect(self.thread3.quit)
            self.worker3.finished.connect(self.worker3.deleteLater)
            self.thread3.finished.connect(self.thread3.deleteLater)
            self.thread3.start()

        else:
            self.thongBao("Bạn chưa detect ra ảnh")

    def event(self):
        self.btnLoadModel.clicked.connect(self.LoadModel)
        self.btnTaiAnh.clicked.connect(self.btnTaiAnhOnclick)
        self.btnDetectImage.clicked.connect(self.NhanDienImage)
        self.btnHienThiInput.clicked.connect(self.btnHienThiInputOnClick)
        self.btnHienThiOutput.clicked.connect(self.btnHienThiOutputOnClick)
        self.btnImageMode.clicked.connect(self.DetectImageMode)
        self.btnVideoMode.clicked.connect(self.DetectVideoMode)
        self.btnTaiVideo.clicked.connect(self.TaiVideo)
        self.btnStartInputVideo.clicked.connect(self.mediaPlayerInput.play)
        self.btnPauseInputVideo.clicked.connect(self.mediaPlayerInput.pause)
        self.btnStopInputVideo.clicked.connect(self.mediaPlayerInput.stop)
        self.btnStartOutputVideo.clicked.connect(self.mediaPlayerOutput.play)
        self.btnPauseOutputVideo.clicked.connect(self.mediaPlayerOutput.pause)
        self.btnStopOutputVideo.clicked.connect(self.mediaPlayerOutput.stop)
        self.btnDetectVideo.clicked.connect(self.NhanDienVideo)
        self.btnHienThiInputVideo.clicked.connect(self.btnHienThiInputVideoOnclick)
        self.btnHienThiOutputVideo.clicked.connect(self.btnHienThiOutputVideoOnclick)