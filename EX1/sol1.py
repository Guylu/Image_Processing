import numpy as np
import matplotlib.pyplot as plt
import skimage.color
import imageio

NUM_PIXEL_VALUE = 255
NUM_PIXEL_ABS = 256
RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]], dtype=np.float64)
RGB = 2
GRAY = 1
Y_CHANNEL = 0
FLOAT64 = 'float64'


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


def imdisplay(filename, representation):
    """
     open a new figure and display the loaded image in the converted
     representation.
    :param filename: the filename of an image on disk
     (could be grayscale or RGB).
    :param representation:- representation code, either 1 or 2 defining whether
    the output should be a grayscale image (1) or an RGB image (2)
    :return:
    """

    img = read_image(filename, representation)
    if representation == GRAY:
        plt.imshow(img, cmap=plt.cm.gray)
    else:
        plt.imshow(img)
    plt.show()


def rgb2yiq(imRGB):
    """
    takes rgb image and return yiq image
    :param imRGB:
    :return:
    """

    return np.dot(imRGB, RGB2YIQ.T.copy())


def yiq2rgb(imYIQ):
    """
     takes yiq image and return rgb image
    :param imYIQ:
    :return:
    """
    return np.dot(imYIQ, np.linalg.inv(RGB2YIQ.T.copy()))


def histogram_equalize(im_orig):
    """
    performs histogram equalization on image
    :param im_orig: input grayscale or RGB float64 image with values in [0, 1]
    :return:a list [im_eq, hist_orig, hist_eq]
    im_eq - is the equalized image. grayscale or RGB float64 image
    with values in [0, 1].
    hist_orig - is a 256 bin histogram of the original image
     (array with shape (256,) ).
    hist_eq - is a 256 bin histogram of the equalized image
     (array with shape (256,) ).

    """
    current_representation = im_orig.ndim - 1
    working_matrix = im_orig
    # rgb check
    if current_representation == RGB:
        pic_yiq = rgb2yiq(im_orig.astype(np.dtype(FLOAT64)))
        working_matrix = pic_yiq[:, :, Y_CHANNEL]
    # calculate histograms
    working_matrix = (working_matrix * NUM_PIXEL_VALUE).round().astype(np.uint8)
    histogram, _ = np.histogram(working_matrix, bins=NUM_PIXEL_ABS,
                                range=(0, NUM_PIXEL_VALUE))
    cumulative = np.cumsum(histogram)

    offset = cumulative[(cumulative != 0).argmin(axis=0)]
    eq = np.vectorize(equalize)
    new_cum = eq(cumulative, offset, cumulative[NUM_PIXEL_VALUE])

    new_pic = new_cum[working_matrix]
    new_hist, _ = np.histogram(new_pic, bins=NUM_PIXEL_ABS,
                               range=(0, NUM_PIXEL_VALUE))

    im_eq = new_pic.astype(np.dtype(FLOAT64)) / NUM_PIXEL_VALUE
    if current_representation == RGB:
        pic_yiq[:, :, Y_CHANNEL] = new_pic.astype(
            np.dtype(FLOAT64)) / NUM_PIXEL_VALUE
        im_eq = yiq2rgb(pic_yiq)
    return [im_eq, histogram, new_hist]


def equalize(a, offset, end):
    """
    
    :param a:
    :param offset:
    :param end:
    :return:
    """
    return np.round(NUM_PIXEL_VALUE * (np.divide(a - offset, end - offset)))


def quantize(im_orig, n_quant, n_iter):
    """
    :param im_orig:the input grayscale or RGB image to be quantized (float64
    image with values in [0, 1]). :param n_quant:the number of intensities
    your output im_quant image should have :param n_iter: the maximum number
    of iterations of the optimization procedure (may converge earlier.)
    :return: The output is a list [im_quant, error] where: im_quant - is the
    quantized output image. (float64 image with values in [0, 1]). error - is
    an array with shape (n_iter,) (or less) of the total intensities error
    for each iteration of the quantization procedure.

    """
    z = np.zeros(n_quant + 1)
    q = np.zeros(n_quant)
    error = []

    current_representation = im_orig.ndim - 1
    working_matrix = im_orig
    # rgb check
    if current_representation == RGB:
        pic_yiq = rgb2yiq(im_orig)
        working_matrix = pic_yiq[:, :, Y_CHANNEL]
    working_matrix = np.round(working_matrix * NUM_PIXEL_VALUE).astype(np.uint8)

    histogram, _ = np.histogram(working_matrix, bins=NUM_PIXEL_ABS,
                                range=(0, NUM_PIXEL_VALUE))
    for i in range(1, n_quant):
        z[i] = z[i - 1] + np.where(np.cumsum(histogram[int(z[i - 1]):]) >
                                   working_matrix.size / n_quant)[0][0]
    z[-1] = NUM_PIXEL_VALUE
    z[0] = -1

    for k in range(n_iter):
        for i in range(n_quant):
            range_to_operate = np.arange(z[i] + 1, z[i + 1] + 1).astype(
                np.int64)
            sum1 = np.inner(histogram[range_to_operate], range_to_operate)
            sum2 = np.sum(histogram[range_to_operate])
            q[i] = sum1 / sum2

        z_prev = z.copy()
        for i in range(1, n_quant):
            z[i] = np.floor((q[i - 1] + q[i]) / 2)

        if np.allclose(z, z_prev):
            break
        # calc error
        e = 0.0
        for i in range(n_quant):
            for g in np.arange(z[i] + 1, z[i + 1] + 1):
                e += (((q[i] - int(g)) ** 2) * histogram[int(g)])
        error.append(e)

    t = np.zeros(NUM_PIXEL_ABS)
    for i in range(n_quant):
        t[int(z[i]):int(z[i + 1])] = q[i]

    new_pic = t[working_matrix]

    im_quant = new_pic.astype(np.dtype(FLOAT64)) / NUM_PIXEL_VALUE
    if current_representation == 2:
        pic_yiq[:, :, Y_CHANNEL] = new_pic.astype(
            np.dtype(FLOAT64)) / NUM_PIXEL_VALUE
        im_quant = yiq2rgb(pic_yiq)
    return [im_quant, error]


if __name__ == '__main__':
    im = read_image("C:/University/Year 3/Semester 1/67829 "
                    "Image "
                    "Processing/Ex's/EX1/low_contrast.jpg", 2)
    # im2 = yiq2rgb(rgb2yiq(im))
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()

    plt.imshow(histogram_equalize(im)[0], cmap=plt.cm.gray)
    plt.show()

    plt.imshow(quantize(histogram_equalize(im)[0],
                        10, 10)[0])
    plt.show()
