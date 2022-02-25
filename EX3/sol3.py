import numpy as np
import matplotlib.pyplot as plt
import skimage.color
import imageio
import scipy.io.wavfile
import imageio
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from scipy import misc
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
from scipy.linalg import pascal

NUM_PIXEL_VALUE = 255
NUM_PIXEL_ABS = 256


def read_image(filename, representation):
    """
    reads an image file and converts it into a given representation
    :param filename: the filename of an image on disk
     (could be grayscale or RGB).
    :param representation:- representation code, either 1 or 2 defining whether
     the output should be a grayscale image (1) or an RGB image (2)
    :return:
    an image, make sure the output image is represented by a matrix of type
    np.float64 with intensities (either grayscale or RGB channel intensities)
    normalized to the range [0, 1]
    """
    img = np.asarray(imageio.imread(filename), dtype=np.float64)
    current_representation = img.ndim - 1
    if current_representation == representation:
        return img / NUM_PIXEL_VALUE

    return skimage.color.rgb2gray(img) / NUM_PIXEL_VALUE


def build_gaussian_pyramid(im, max_levels, filter_size):
    """

    :param im:a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the representation set to 1).
    :param max_levels:the maximal number of levels1 in the resulting pyramid
    :param filter_size:the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
    in constructing the pyramid filter
    :return:pyr, filter_vec
    """
    pyr = [im]
    working_im = im.copy()
    kernel = np.zeros(2)
    kernel[0] = 1
    kernel[1] = 1
    final_kernel = np.zeros(filter_size)
    final_kernel[0] = 1
    final_kernel[1] = 1
    for i in range(filter_size - 2):
        final_kernel = convolve(final_kernel, kernel, mode="constant", origin=-1)
    final_kernel = final_kernel.reshape(1, -1) / np.sum(final_kernel)
    max_levels -= 1
    while max_levels > 0 and pyr[-1].shape[0] > 16 and pyr[-1].shape[1] > 16:
        working_im = _calc_pyr_level(final_kernel, working_im)
        pyr.append(working_im)
        max_levels -= 1
    return pyr, final_kernel


def _calc_pyr_level(kernel, working_im):
    blurred_x = convolve(working_im, kernel)
    blurred_y1 = convolve(blurred_x, kernel)
    blurred_y = convolve(blurred_x, np.transpose(kernel))
    # res = blurred_y[:, np.arange(0, blurred_y.shape[1] - 1, 2)]
    # res = res[np.arange(0, res.shape[0] - 1, 2)]
    res = blurred_y[::2, ::2]
    return res


def expand(level, kernel):
    pad = (2) * np.array(level.shape)
    expanded = np.zeros(pad, dtype=level.dtype)
    expanded[::2, ::2] = level
    blurred = convolve(expanded, kernel)
    fin = convolve(blurred, kernel.T)
    return fin


def build_laplacian_pyramid(im, max_levels, filter_size):
    """

    :param im:a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the representation set to 1).
    :param max_levels:the maximal number of levels1 in the resulting pyramid
    :param filter_size:the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
    in constructing the pyramid filter
    :return:pyr, filter_vec
    """
    pyr = []
    g, final_kernel = build_gaussian_pyramid(im, max_levels, filter_size)
    final_kernel *= 2
    for i in range(len(g) - 1):
        shrunk = g[i + 1]
        fin = expand(shrunk, final_kernel)
        pyr.append(g[i] - fin)
    pyr.append(g[-1])
    return pyr, final_kernel / 2


def laplacian_to_image(lpyr, filter_vec, coeff):
    res = lpyr[-1] * coeff[-1]
    for i in range(len(lpyr) - 1, 0, -1):
        res = expand(res, 2 * filter_vec) + lpyr[i - 1] * coeff[i - 1]
    return res


def render_pyramid(pyr, levels):
    res = np.zeros((pyr[0].shape[0], int(pyr[0].shape[1] * (0.5 ** levels - 1) / -0.5)))
    res[0:pyr[0].shape[0], 0:pyr[0].shape[1]] = pyr[0]
    sum2 = 0
    for i in range(levels):
        res[0:pyr[i].shape[0], sum2:sum2 + pyr[i].shape[1]] = np.interp(
            pyr[i], (pyr[i].min(), pyr[i].max()), (0, 1))
        sum2 += pyr[i].shape[1]
    return res


def display_pyramid(pyr, levels):
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap="gray")
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    l1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    ker = l1[1]
    l1 = l1[0]
    l2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    gm = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[0]

    l_out = [np.zeros_like(a) for a in l1]
    for i in range(len(l1)):
        l_out[i] = gm[i] * l1[i] + (1 - gm[i]) * l2[i]
    return laplacian_to_image(l_out, ker, np.ones(len(im1[0])))


def make_me_boolean(img):
    max_val = np.max(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (img[i, j] > max_val / 1.5)
    return img


import os


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    im1 = read_image(relpath("s1_new.jpg"), 2)
    im2 = read_image(relpath("p1p3.jpg"), 2)
    mask = make_me_boolean(read_image(relpath("m1.jpg"), 1))
    k = 189
    res = np.zeros((im1.shape[0], im1.shape[1], 3))
    for i in range(3):
        res[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, k, k, k)
    fig, axarr = plt.subplots(2, 2)
    mask = np.array(mask, dtype=bool)
    axarr[0, 0].imshow(im1)
    axarr[0, 1].imshow(im2)
    axarr[1, 0].imshow(mask, cmap="gray")
    axarr[1, 1].imshow(res)
    plt.show()
    return im1, im2, mask, res


def blending_example2():
    im2 = read_image(relpath("backk.jpg"), 2)
    im1 = read_image(relpath("frontt.jpg"), 2)
    mask = make_me_boolean(read_image(relpath("maskk.jpg"), 1))
    res = np.zeros((im1.shape[0], im1.shape[1], 3))
    for i in range(3):
        res[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, 4, 3, 3)
    fig, axarr = plt.subplots(2, 2)
    mask = np.array(mask, dtype=bool)
    axarr[0, 0].imshow(im1)
    axarr[0, 1].imshow(im2)
    axarr[1, 0].imshow(mask, cmap="gray")
    axarr[1, 1].imshow(res)
    plt.show()
    return im1, im2, mask, res


def _test_pyr():
    im = read_image("s1_new.jpg", 1)

    gpyr, _ = build_gaussian_pyramid(im, 3, 3)
    lpyr, gfilter = build_laplacian_pyramid(im, 3, 3)

    display_pyramid(gpyr, len(gpyr))
    display_pyramid(lpyr, len(lpyr))


def _test_lap_to_im():
    im = read_image("s1_new.jpg", 1)

    lpyr, gfilter = build_laplacian_pyramid(im, 3, 3)

    res = laplacian_to_image(lpyr, gfilter, [1, 1, 2])
    plt.imshow(res, cmap="gray")
    plt.show()
    plt.imshow(im, cmap="gray")
    plt.show()


if __name__ == '__main__':
    # _test_pyr()
    # _test_lap_to_im()
    blending_example1()
    blending_example2()
    # im1, im2, mask, im_blend = blending_example1()
    # plt.imshow(im_blend, cmap="gray")
    # plt.show()
    # im1 = read_image("presubmit_externals/monkey.jpg", 1)
    # im2 = read_image("presubmit_externals/front.jpg", 1)
    # mask = read_image("presubmit_externals/mask.jpg", 1)

    # plt.imshow(im, cmap="gray")
    # plt.show()
    # g = build_gaussian_pyramid(im, 5, 7)
    # m = np.zeros_like(g[0])
    # for i in g[0]:
    #     plt.imshow(i, cmap="gray")
    #     plt.show()
    # im = laplacian_to_image(g[0], g[1], np.ones(len(g[0])))
    # plt.imshow(im, cmap="gray")
    # plt.show()
    # display_pyramid(g[0], len(g[0]))
    # mask = np.zeros_like(im)
    # mask[0:im.shape[0], 0:im.shape[1] // 2] = np.ones((im.shape[0], im.shape[1] // 2))
    # im_blend = pyramid_blending(im1, im2, mask, 5, 7, 7)
    # display_pyramid(build_laplacian_pyramid(im1, 6, 3)[0], 6)
    # plt.imshow(im_blend, cmap="gray")
    # plt.show()
