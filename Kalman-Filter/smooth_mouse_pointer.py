import sys
import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QColor
from scipy.stats import multivariate_normal as mvn

from kalman_filter import KalmanFilter

class SmoothMousePointer(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(1000, 1000)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)

        self.last_x, self.last_y = None, None
        self.last_Vx, self.last_Vy = None, None
        self.dt = 0.1

        self.kf = KalmanFilter.create_filter(self.dt)

        np.random.seed(12345)

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            self.last_Vx = 0
            self.last_Vy = 0
            return # Ignore the first time.

        Vx = (e.x() - self.last_x) / self.dt
        Vy = (e.y() - self.last_y) / self.dt

        xt = self._estimate_x()
        yt = self._estimate_y(xt)

        self.kf.predict()
        self.kf.update(yt)
        self.kf.log()

        x_pred = self.kf.log_x[-1][0]
        y_pred = self.kf.log_x[-1][1]

        painter = QtGui.QPainter(self.label.pixmap())
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()


        painter_pred = QtGui.QPainter(self.label.pixmap())
        painter_pred.setPen(QColor(255, 0, 0))
        painter_pred.drawEllipse(QPoint(x_pred, y_pred), 1, 1)
        painter_pred.end()

        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()
        self.last_Vx = Vx
        self.last_Vy = Vy

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None


    def _estimate_x(self):
        x_prev = np.array([self.last_x, self.last_y, self.last_Vx, self.last_Vy])
        x = self.kf.A.dot(x_prev) + mvn.rvs(cov=self.kf.Q)
        return x

    def _estimate_y(self, x):
        y = self.kf.H.dot(x) + mvn.rvs(cov=self.kf.R)
        return y


app = QtWidgets.QApplication(sys.argv)
window = SmoothMousePointer()
window.show()
app.exec_()
