import unittest
from sol1 import *
import skimage.exposure as ex
import skimage.color as sk


MAX_NORMED_VALUE = 1.001
HIST_EPSILON = 0.05
TRANS_EPSILON = 0.001
IMGS = ["jerusalem.jpg", "low_contrast.jpg",
        "monkey.jpg"]


class MyTestCase(unittest.TestCase):
    def is_normed(self, image):
        return (image<=MAX_NORMED_VALUE).all() and image.dtype == np.float64

    def test_read(self):
        for fname in IMGS:
            print("-- Reading: ", fname)
            im = read_image(fname,1)
            assert self.is_normed(im)
            assert im.ndim == 2
            print("---- As gray: OK")
            im = read_image(fname,2)
            assert self.is_normed(im)
            assert im.ndim == 3
            print("---- As rgb: OK")


    def test_r2y_transform(self):
        for fname in IMGS:
            print("-- Transforming to YIQ: ", fname)
            im = read_image(fname, 2)

            yiq = rgb2yiq(im)
            ex_yiq = sk.rgb2yiq(im)
            error = np.abs(yiq-ex_yiq)

            print(" ---- max error: ", error.max())
            assert self.is_normed(im)
            assert (error < TRANS_EPSILON).all()
            print(" ---- to yiq OK")

            print("-- Transforming back to RGB: ", fname)
            rgb = yiq2rgb(yiq)
            ex_rgb = sk.yiq2rgb(yiq)
            error = np.abs(rgb-ex_rgb)

            print(" ---- max error: ", error.max())
            print((rgb<1.01).all())
            assert self.is_normed(rgb)
            assert (error < TRANS_EPSILON).all()
            print(" ---- max diff from original: ", (im-rgb).max())
            print(" ---- to rgb OK")

    def test_eq_hist_g(self):
        for fname in IMGS:
            print("-- Equalizing grayscale: ", fname)
            im = read_image(fname, 1)
            eq_img, orig_hist, eq_hist = histogram_equalize(im)
            ex_img = ex.equalize_hist(im)
            error = np.abs(eq_img-ex_img)

            im_err = (error < HIST_EPSILON).all()
            print("---- max error: "+ str(np.abs((eq_img -
                                                  im)).max()))
            # assert self.is_normed(eq_img)
            assert im_err
            print("---- OK")

    def test_eq_hist_rgb(self):
        for fname in IMGS:
            print("-- Equalizing rgb: ", fname)
            ex_im = read_image(fname, 2)
            eq_img, orig_hist, eq_hist = histogram_equalize(ex_im)
            ex_im = sk.rgb2yiq(ex_im)
            ex_im[:,:,0] = ex.equalize_hist(ex_im[:,:,0])
            ex_im = sk.yiq2rgb(ex_im)

            print("---- max error: " + str(np.abs((
                    eq_img-ex_im)).max()))
            im_err = (np.abs((eq_img-ex_im)) < HIST_EPSILON).all()
            # assert self.is_normed(eq_img)
            assert(im_err)
            print("---- OK")


if __name__ == '__main__':
    unittest.main()
