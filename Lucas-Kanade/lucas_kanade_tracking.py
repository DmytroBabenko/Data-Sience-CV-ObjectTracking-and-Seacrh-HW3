import argparse
import cv2 as cv
import numpy as np

from utils import Utils
# from lucas_kanade import LucasKanade
from lucas_kanade import LucasKanade, LucasKanadeFillRectPoints, LucasKanadeGoodFeatures

class Tracker:

    def __init__(self):
        self.lucas_kanade_tracker = LucasKanadeFillRectPoints() #LucasKanadeGoodFeatures

    def run(self, images, init_rect, display_fun):

        rect = init_rect
        for i in range(1, len(images)):
            prev_img = images[i-1]
            img = images[i]

            rect = self.lucas_kanade_tracker.track(prev_img, img, rect)
            display_fun(img, tuple(rect))

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break


def display_function(img, rect):
    rect_immg = Utils.draw_rect_window_on_image(img, rect, (255, 0, 0))
    cv.imshow("Lucas Kanade tracking", rect_immg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', dest="dataset_dir", type=str, help='images directory')
    parser.add_argument('--rect', dest='rect', type=str, help="initial rectangle")

    args = parser.parse_args()

    images = Utils.read_images_from_dir(f"{args.dataset_dir}")
    rect = [int(val) for val in args.rect.split(',')]
    tracker = Tracker()
    tracker.run(images, rect, display_function)

if __name__ == '__main__':
    main()
