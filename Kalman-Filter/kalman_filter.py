import numpy as np

#this filter is used from bayesian statistics course:
# https://gitlab.com/kamil.dedecius/ucu-bm/blob/master/zdrojaky/kf/kf.py
#Theoretical part are here: https://gitlab.com/kamil.dedecius/ucu-bm/blob/master/lectures/4_lecture.ipynb

class KalmanFilter():

    @staticmethod
    def create_filter(dt=0.1):
        q = 2
        r = 10
        #state transition model
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        #covariance of the process noise
        Q = q * np.array(
            [[(dt ** 3) / 3, 0, (dt ** 2) / 2, 0],
             [0, (dt ** 3) / 3, 0, (dt ** 2) / 2],
             [(dt ** 2) / 2, 0, dt, 0],
             [0, (dt ** 2) / 2, 0, dt]])

        #observation model
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])


        #covariance of the observation noise
        R = (r ** 2) * np.array([[1, 0],
                                 [0, 1]])

        return KalmanFilter(A, None, H, R, Q)



    def __init__(self, A, B, H, R, Q):
        self.A = A
        self.B = B
        #        self.H = H
        self.H = np.atleast_2d(H)
        self.Q = Q
        self.P = np.eye(A.shape[0]) * 1000.
        self.x = np.zeros(A.shape[0])
        self.log_x = []
        self.xi = np.zeros(np.asarray(self.P.shape) + 1)
        self.R = R
        if np.isscalar(R):
            self.Rinv = 1 / R
        else:
            self.Rinv = np.linalg.inv(R)

    def predict(self, u=None):
        xminus = self.A.dot(self.x)
        if u is not None:
            xminus += self.B.dot(u)
        Pminus = self.A.dot(self.P).dot(self.A.T) + self.Q
        xi_vector = np.r_[xminus[np.newaxis], np.eye(self.x.shape[0])]
        self.xi = xi_vector.dot(np.linalg.inv(Pminus)).dot(xi_vector.T)
        self.x = xminus  # temporary
        self.P = Pminus  # temporary

    def update(self, y, Rinv=None):
        if Rinv is None:
            Rinv = self.Rinv
        y = np.atleast_2d(y).reshape(self.H.shape[0], -1)
        T_vector = np.concatenate((y, self.H), axis=1)
        T = T_vector.T.dot(Rinv).dot(T_vector)
        self.xi += T
        self.P = np.linalg.inv(self.xi[1:, 1:])
        self.x = self.P.dot(self.xi[1:, 0])

    def log(self):
        self.log_x.append(self.x.copy())
