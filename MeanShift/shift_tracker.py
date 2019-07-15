import glob
from abc import abstractmethod

import numpy as np
import cv2 as cv
import argparse

class Tracker:

    def __init__(self):
        self.term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        self.lowerb = np.array((0., 60., 32.))
        self.uperb = np.array((180., 255., 255.))
        self.channles_hist = [2]
        self.hist_size = [180]
        self.hits_ranges = [0, 180]

        self.red_color = (255, 0, 0)

    @staticmethod
    def create(method):
        if method == "meanshift":
            return MeanShift()
        elif method == "camshift":
            return CamShift()

        return None

    @abstractmethod
    def run(self, images, x, y, w, h, display_fun=None): pass



    def _create_roi_hist_for_tracker(self, init_img, x, y, w, h):
        roi = init_img[y:y + h, x:x + w]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, self.lowerb, self.uperb)
        roi_hist = cv.calcHist([hsv_roi], self.channles_hist, mask, self.hist_size, self.hits_ranges)
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        return roi_hist


    @staticmethod
    def draw_rect_window_on_image(image, rect, color):
        x, y, w, h = rect
        img = cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
        return img


    @staticmethod
    def draw_contour_window_on_image(image, contour_pos, color):
        pts = cv.boxPoints(contour_pos)
        pts = np.int0(pts)
        img = cv.polylines(image,[pts],True, color,2)
        return img




class MeanShift(Tracker):

    def __init__(self):
        Tracker.__init__(self)

    def run(self, images, x, y, w, h, display_fun=None):
        if len(images) == 0:
            return []

        init_img = images[0]
        roi_hist = self._create_roi_hist_for_tracker(init_img, x, y, w, h)
        track_window = (x, y, w, h)


        tracked_images = [Tracker.draw_rect_window_on_image(init_img, track_window, self.red_color)]

        for i in range(1, len(images)):
            image = images[i]

            ret, track_window = self._track(image, track_window, roi_hist)

            tracked_img = Tracker.draw_rect_window_on_image(image, track_window, self.red_color)

            if display_fun is not None:
                display_fun(tracked_img)

            tracked_images.append(tracked_img)

        return tracked_images


    def _track(self, image, init_track_window, roi_hist):

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],self.channles_hist,roi_hist, self.hits_ranges,1)

        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, init_track_window, self.term_crit)

        return ret, track_window


class CamShift(Tracker):

    def __init__(self):
        Tracker.__init__(self)


    def run(self, images, x, y, w, h, display_fun=None):
        if len(images) == 0:
            return []

        init_img = images[0]
        roi_hist = self._create_roi_hist_for_tracker(init_img, x, y, w, h)
        track_window = (x, y, w, h)


        tracked_images = [Tracker.draw_rect_window_on_image(init_img, track_window, self.red_color)]

        for i in range(1, len(images)):
            image = images[i]

            tracked_pos, track_window = self._track(image, track_window, roi_hist)

            tracked_img = Tracker.draw_contour_window_on_image(image, tracked_pos, self.red_color)

            if display_fun is not None:
                display_fun(tracked_img)

            tracked_images.append(tracked_img)

        return tracked_images


    def _track(self, image, init_track_window, roi_hist):

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],self.channles_hist,roi_hist, self.hits_ranges,1)

        # apply meanshift to get the new location
        ret, track_window = cv.CamShift(dst, init_track_window, self.term_crit)

        return ret, track_window


    @staticmethod
    def draw_track_window_on_img(image, track_pos, color):
        pts = cv.boxPoints(track_pos)
        pts = np.int0(pts)
        img = cv.polylines(image,[pts],True, color,2)
        return img


def get_images_from_dir(dir):
    filenames = [img for img in glob.glob(f"{dir}/*.jpg")]
    filenames.sort()

    images = []
    for file in filenames:
        img = cv.imread(file)
        images.append(img)

    return images

def read_ground_truth_rect(file):
    rects = []
    with open(file) as f:
        while True:
            # read line
            line = f.readline()
            if not line:
                break
            if ',' in line:
                rect = tuple([int(val) for val in line.split(',')])
                rects.append(rect)
            elif '\t' in line:
                rect = tuple([int(val) for val in line.split('\t')])
                rects.append(rect)


    return rects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', dest="dataset_dir", type=str, help='images directory')
    parser.add_argument('--rect', dest='rect', type=str, help="initial rectangle")
    parser.add_argument('--method', dest='method', type=str, help="meanshift|camshift")

    args = parser.parse_args()

    images = get_images_from_dir(f"{args.dataset_dir}/img")
    ground_truth_rects = read_ground_truth_rect(f"{args.dataset_dir}/groundtruth_rect.txt")
    x, y, w, h = ground_truth_rects[0]

    tracker = Tracker.create(args.method)
    if tracker is None:
        print(f"Unknown method {args.method}")
        exit(1)

    tracked_images = tracker.run(images, x, y, w, h)

    for i in range(0, len(tracked_images)):
        tracked_img = tracked_images[i]
        Tracker.draw_rect_window_on_image(tracked_img, ground_truth_rects[i], (0, 255, 0))
        cv.imshow("window", tracked_img)
        k = cv.waitKey(50) & 0xff
        if k == 27:
            break


if __name__ == "__main__":
    main()