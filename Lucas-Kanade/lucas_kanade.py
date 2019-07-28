from abc import abstractmethod

import cv2 as cv
import numpy as np


class LucasKanade:
    def __init__(self):
        pass

    @abstractmethod
    def track(self, prev_img, img, rect): pass

    @staticmethod
    def calc_delta(prev_points, points):
        assert len(prev_points) == len(points)

        if len(points) == 0:
            return 0, 0

        delta_x = points[:, 0] - prev_points[:, 0]
        delta_y = points[:, 1] - prev_points[:, 1]

        dx = np.mean(delta_x)
        dy = np.mean(delta_y)

        return dx, dy


class LucasKanadeGoodFeatures(LucasKanade):

    def __init__(self):
        super().__init__()
        self.lk_winSize = (10, 10)
        self.lk_maxLevel = 2
        self.lk_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 0.01)

        self.lk_params = dict( winSize=self.lk_winSize,
                               maxLevel=self.lk_maxLevel,
                               criteria=self.lk_criteria)

        self.feature_params = dict(maxCorners = 2000,
                       qualityLevel = 0.5,
                       minDistance = 3,
                       blockSize = 12,
                       useHarrisDetector = True,
                       k = 0.1)



    def track(self, prev_img, img, rect):

        prev_gray = cv.cvtColor(prev_img, cv.COLOR_BGR2GRAY)

        # Equalize the grey histogram to minimize lighting effects
        prev_gray = cv.equalizeHist(prev_gray)

        feature_points = self._get_feature_points(prev_gray, rect)
        #
        # _, prev_pyramid = cv.buildOpticalFlowPyramid(prev_img, winSize=self.lk_winSize, maxLevel=self.lk_maxLevel)
        # _, pyramid = cv.buildOpticalFlowPyramid(img, winSize=self.lk_winSize, maxLevel=self.lk_maxLevel)

        points, st, err = cv.calcOpticalFlowPyrLK(prev_img, img, feature_points, None, **self.lk_params)

        width, height = tuple(img.shape[:2])

        prev_points, points = self._filter_prev_and_cur_points_out_of_rect(feature_points, points, (0, 0, width, height))
        # points = self._filter_points_out_of_rect(points, (0, 0, width, height))

        dx, dy = LucasKanade.calc_delta(prev_points, points)

        rect[0] += int(dx)
        rect[1] += int(dy)

        return rect



    def _get_feature_points(self, image, rect):
        x, y, w, h = rect

        mask = np.zeros_like(image)
        mask[y:y+h, x:x+w] = 255

        points = cv.goodFeaturesToTrack(image, mask=mask, **self.feature_params)

        if points is None or len(points) == 0:
            return self._get_common_rect_points(rect)

        return self._filter_points_out_of_rect(points.reshape(-1, 2), rect)


    def _filter_points_out_of_rect(self, points, rect):
        x, y, w, h = rect

        filtered_points = filter(lambda item: item[0] >= x and item[0] <= x + w and item[1] >= y and item[1] <= y+h,
                                 points)

        return np.array(list(filtered_points))


    def _filter_prev_and_cur_points_out_of_rect(self, prev_points, points, rect):
        x, y, w, h = rect

        all_points = list(zip(prev_points, points))

        filtered = filter(lambda item: item[1][0] >= x and item[1][0] <= x + w and item[1][1] >= y and item[1][1] <= y+h,
                                 all_points)

        l = list(zip(*list(filtered)))

        if len(l) == 0:
            return np.array([]), np.array([])

        prev = np.array(l[0])
        cur = np.array(l[1])

        return prev, cur



    def _get_common_rect_points(self, rect):
        x, y, w, h = rect

        points = []

        #corner points
        points.append([x, y])
        points.append([x, y+h])
        points.append([x+w, y])
        points.append([x+w, y+h])

        #center point
        points.append([x + w//2, y + h//2])

        #quater points
        points.append([x + w//4, y + w//4])
        points.append([x + w//4, y + 3*w//4])
        points.append([x + 3*w//4, y + w//4])
        points.append([x + 3*w//4, y + 3*w//4])

        return np.array(points, dtype=np.float32)


class LucasKanadeFillRectPoints(LucasKanade):

    def __init__(self):
        super().__init__()
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    def track(self, prev_img, img, rect):

        (x, y, w, h) = rect

        rect_points = []
        for i in range(0, h):
            rect_points += [[x + j, y+i] for j in range(0, w)]

        rect_points = np.array(rect_points, dtype=np.float32)

        next_points, st, err = cv.calcOpticalFlowPyrLK(prev_img, img, rect_points, None, **self.lk_params)

        dx, dy = LucasKanade.calc_delta(rect_points, next_points)

        rect[0] += int(dx)
        rect[1] += int(dy)

        return rect






















