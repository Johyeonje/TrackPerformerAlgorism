from object_tracker_cus2 import Apply_Models

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

running = False
capture_thread = None
form_class = uic.loadUiType("simple3.ui")[0]
q = queue.Queue()
state = 0
start_X = 0
width_X = 1920
start_Y = 0
height_Y = 1080


def grab(cam, queue, width, height, fps):
    global running
    capture = cv2.VideoCapture(cam)
    width_X = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_Y = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # capture.set(cv2.CAP_PROP_FPS, fps)

    apply = Apply_Models()
    reset = 1

    while running:
        frame = {}
        capture.grab()
        retval, img = capture.retrieve(0)
        img = img[int(start_Y):int(height_Y)+1, int(start_X):int(width_X)+1]
        # cv2.imshow('pyqt1', img)

        if state == 1:
            if reset == 1:
                # print(start_X, width_X, start_Y, height_Y)
                apply.set_tracker()
                reset = 0
            img = apply.main(img)
        else:
            reset = 1

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
        self.offButton.setChecked(True)
        self.onButton.clicked.connect(self.on_button_clicked)
        self.offButton.clicked.connect(self.off_button_clicked)

        self.Loc_apply.clicked.connect(self.get_editText)

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
        self.x_start.setText('0')
        self.x_end.setText('1920')
        self.y_start.setText('0')
        self.y_end.setText('1080')

    def on_button_clicked(self):
        global state
        state = 1

    def off_button_clicked(self):
        global state
        state = 0

    def get_editText(self):

        global start_X
        global width_X
        global start_Y
        global height_Y

        if self.x_start.text():
            start_X = self.x_start.text()
        else:
            self.x_start.setText(start_X)

        if self.x_end.text():
            width_X = self.x_end.text()
        else:
            self.x_end.setText(width_X)

        if self.y_start.text():
            start_Y = self.y_start.text()
        else:
            self.y_start.setText(start_Y)

        if self.y_end.text():
            height_Y = self.y_end.text()
        else:
            self.y_end.setText(height_Y)

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

capture_thread = threading.Thread(target=grab, args=('./data/video/tt.mp4', q, 1920, 1080, 10))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindowClass(None)
    w.setWindowTitle('Performer Tracking Program Prototype v2.7')
    w.show()
    app.exec_()