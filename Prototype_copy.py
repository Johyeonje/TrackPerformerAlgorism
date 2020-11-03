# -*- coding: utf-8 -*-

__author__ = "Saulius Lukse"
__copyright__ = "Copyright 2016, kurokesu.com"
__version__ = "0.1"
__license__ = "GPL"

import numpy as np
import tensorflow as tf
from deep_sort import nn_matching
from object_tracker_cus6 import Apply_Models
from deep_sort.tracker import Tracker

import queue
import sys
import threading

import cv2
from PyQt5 import uic
from PyQt5.QtCore import Qt, QUrl, QSize, QPoint, QTimer
from PyQt5.QtGui import QIcon, QFont, QPainter, QImage
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QPushButton, QSlider, QStyle, QVBoxLayout, QWidget,
                             QStatusBar, QMainWindow,
                             QAction, qApp)

from tensorflow.python.saved_model import tag_constants

running = False
capture_thread = None
form_class = uic.loadUiType("simple.ui")[0]
q = queue.Queue()


def grab(cam, queue, width, height, fps):
    global running

    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    # s_fps = int(capture.get(cv2.CAP_PROP_FPS))
    s_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    s_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # frame_size = (width, height)
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('./outputs/demoF.avi', codec, 20, (s_width, s_height))

    apply = Apply_Models()


    # while running:
    while True:
        frame = {}
        capture.grab()
        retval, img = capture.retrieve(0)
        # cv2.imshow('pyqt1', img)

        img = apply.main(img)
        out.write(img)
        frame["img"] = img

        if queue.qsize() < 10:
            queue.put(frame)
        else:
            print
            queue.qsize()


class OwnImageWidget(QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):  # 이 코드에 Yolov4 적용하면 될것으로 보임
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QPainter()

        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()


class MyWindowClass(QMainWindow, form_class):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.initUI()

        self.startButton.clicked.connect(self.start_clicked)

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def initUI(self):
        camAction = QAction('Use Cam', self)
        camAction.setShortcut('Ctrl+C')
        camAction.setStatusTip('Use Cam')
        camAction.triggered.connect(self.start_clicked)

        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        self.statusBar()

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(camAction)
        filemenu.addAction(exitAction)


    def start_clicked(self):
        global running
        running = True
        capture_thread.start()
        self.startButton.setEnabled(False)
        self.startButton.setText('Starting...')

    def update_frame(self):
        if not q.empty():
            self.startButton.setText('Camera is live')
            frame = q.get()
            img = frame["img"]
            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1

            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QImage(img.data, width, height, bpl, QImage.Format_RGB888)
            self.ImgWidget.setImage(image)

    def closeEvent(self, event):
        global running
        running = False


capture_thread = threading.Thread(target=grab, args=('./data/video/tttt.mp4', q, 1920, 1080, 16))

if __name__ == '__main__':

    app = QApplication(sys.argv)
    w = MyWindowClass(None)
    w.setWindowTitle('Kurokesu PyQT OpenCV USB camera test panel')
    w.show()
    app.exec_()
