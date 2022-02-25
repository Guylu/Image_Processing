from scipy.signal import convolve2d
import numpy as np
from scipy.ndimage.filters import convolve
import imageio
import skimage.color


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


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
    """

    :param kernel:
    :param working_im:
    :return:
    """
    blurred_x = convolve(working_im, kernel)
    blurred_y = convolve(blurred_x, np.transpose(kernel))
    res = blurred_y[::2, ::2]
    return res


def expand(level, kernel):
    """

    :param level:
    :param kernel:
    :return:
    """
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
    """

    :param lpyr:
    :param filter_vec:
    :param coeff:
    :return:
    """
    res = lpyr[-1] * coeff[-1]
    for i in range(len(lpyr) - 1, 0, -1):
        res = expand(res, 2 * filter_vec) + lpyr[i - 1] * coeff[i - 1]
    return res


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """

    :param im1:
    :param im2:
    :param mask:
    :param max_levels:
    :param filter_size_im:
    :param filter_size_mask:
    :return:
    """
    l1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    ker = l1[1]
    l1 = l1[0]
    l2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    gm = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[0]

    l_out = [np.zeros_like(a) for a in l1]
    for i in range(len(l1)):
        l_out[i] = gm[i] * l1[i] + (1 - gm[i]) * l2[i]
    return laplacian_to_image(l_out, ker, np.ones(len(im1[0])))
