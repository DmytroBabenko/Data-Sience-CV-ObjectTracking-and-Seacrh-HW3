import argparse
import numpy as np
import cv2 as cv

from utils import Utils


class TemplateMatching:

    def __init__(self):
        self.method_dict = self._create_method_dict()


    @staticmethod
    def _check_correctness(image : np.ndarray, template: np.ndarray):
        assert image.ndim == template.ndim

        H, W = image.shape[:2]
        h, w = template.shape[:2]

        assert h < H
        assert w < W


    def match(self, image, template, method='ssd'):
        if method not in self.method_dict:
            raise Exception(f"Unknown method {method}")

        R = self._match(image, template, self.method_dict[method])
        h, w = template.shape[:2]

        xy = [0, 0]
        if method == 'ncc':
            xy = np.argmax(R, axis=1)
        else:
            xy = np.argmin(R, axis=1)

        x, y = xy[0], xy[1]

        rect = (x, y, w, h)

        return rect

    def _match(self, image, template, op_fun):

        self._check_correctness(image, template)

        H, W = image.shape[:2]
        h, w = template.shape[:2]

        W_new = W - w + 1
        H_new = H - h + 1

        R = np.zeros((H_new, W_new))

        for y in range(0, H_new):
            for x in range(0, W_new):
                sub_img = image[y:y+h, x:x+w]
                val = op_fun(sub_img, template)
                R[y, x] = val

        return R



    def _ssd(self, img1, img2):
        diff = img2 - img1
        square_diff = diff ** 2
        result = np.sum(square_diff)
        return result


    def _sad(self, img1, img2):
        abs_diff = np.abs(img2 - img1)
        result = np.sum(abs_diff)
        return result


    def _ncc(self, img1, img2):
        x1 = self._normalize(img1)
        x2 = self._normalize(img2)

        result = np.sum(x1 * x2)
        return result



    def _normalize(self, x):
        mean = np.mean(x)
        std = np.std(x)

        x_normalized = (x - mean) / std

        return x_normalized


    def _create_method_dict(self):
        method_dict = {}
        method_dict['ssd'] = self._ssd
        method_dict['sad'] = self._sad
        method_dict['ncc'] = self._ncc

        return method_dict



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest="image", type=str, help='image file')
    parser.add_argument('--template', dest='template', type=str, help="template file")
    parser.add_argument('--method', dest='method', type=str, help="template matching method (ssd|sad|ncc)")
    parser.add_argument('--output', dest='output', type=str, help='output file to write the result')

    args = parser.parse_args()

    image = cv.imread(args.image)
    template = cv.imread(args.template)

    template_macthing = TemplateMatching()

    rect = template_macthing.match(image, template, args.method)

    img_rect = Utils.draw_rect_window_on_image(image, rect, (255, 0, 0))

    cv.imwrite(args.output, img_rect)



if __name__ == '__main__':
    main()

