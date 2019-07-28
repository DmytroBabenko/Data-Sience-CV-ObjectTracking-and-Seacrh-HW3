import glob
import cv2 as cv

class Utils:

    @staticmethod
    def read_images_from_dir(dir):
        filenames = [img for img in glob.glob(f"{dir}/*.jpg")]
        filenames.sort()

        images = []
        for file in filenames:
            img = cv.imread(file)
            # img = cv.imread(file, cv.IMREAD_GRAYSCALE)
            images.append(img)

        return images


    @staticmethod
    def draw_rect_window_on_image(image, rect, color):
        x, y, w, h = rect
        img = cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
        return img
