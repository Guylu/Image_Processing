import numpy as np
import matplotlib.pyplot as plt
import skimage.color
import imageio
import scipy.io.wavfile
import imageio
import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc
from ex2_presubmit import ex2_helper


def DFT(signal):
    """

    :param signal:an array of dtype float64 with shape (N,1)
    :return: DFT of signal
    """
    size = signal.shape[0]
    range_to_operate = np.arange(size).reshape((1, size))
    return np.dot(
        np.exp(-2 * np.pi * 1j * range_to_operate.T * range_to_operate / size),
        signal)


def IDFT(fourier_signal):
    """

    :param fourier_signal:an array of dtype complex128 with the same shape.
    :return: inverse DFT of fourier_signal
    """
    size = fourier_signal.shape[0]
    range_to_operate = np.arange(size).reshape((1, size))
    return np.dot(
        np.exp(2 * np.pi * 1j * range_to_operate.T * range_to_operate / size),
        fourier_signal) / size


def DFT2(image):
    """

    :param image: a grayscale image of dtype float64
    :return:
    """
    DFT_rows = np.apply_along_axis(DFT, 0, image)
    DFT_all = np.apply_along_axis(DFT, 1, DFT_rows)
    return DFT_all


def IDFT2(fourier_signal):
    IDFT_rows = np.apply_along_axis(IDFT, 0, fourier_signal)
    IDFT_all = np.apply_along_axis(IDFT, 1, IDFT_rows)
    return IDFT_all


def _read_file(filename):
    return scipy.io.wavfile.read(filename)


def _save_file(filename, data, rate):
    scipy.io.wavfile.write(filename, rate, data)


def change_rate(filename, ratio):
    rate, data = _read_file(filename)
    _save_file("change_rate.wav", data, int(rate * ratio))


def change_samples(filename, ratio):
    rate, data = _read_file(filename)
    data_new = resize(data, ratio).astype(data.dtype)
    _save_file("change_samples.wav", data_new, rate)


def resize(data, ratio):
    data_dft = DFT(data)
    shifted = np.fft.fftshift(data_dft)
    n = len(shifted)
    if ratio >= 1:
        cropped = shifted[int((n - n * 1 / ratio) // 2):
                          -int((n - n * 1 / ratio) // 2)]
        res = cropped
    else:
        # ratio < 1
        padded = np.pad(shifted, (int(np.floor(n * (ratio))),
                                  int(np.ceil(n * (ratio)))),
                        'constant', constant_values=(0, 0))
        res = padded
    res = np.fft.ifftshift(res)
    return IDFT(res)


def resize_spectrogram(data, ratio):
    data_stft = ex2_helper.stft(data)
    stack = np.zeros(int(np.ceil((data_stft.shape[1] / ratio))))
    for row in data_stft:
        new_line = resize(row.T, ratio).T
        stack = np.vstack((stack, new_line))
    stack = stack[1:stack.shape[0], :]
    return ex2_helper.istft(stack).astype(data.dtype)


def resize_vocoder(data, ratio):
    data_stft = ex2_helper.stft(data)
    stack = np.zeros(int(np.ceil((data_stft.shape[1] / ratio))))
    for row in data_stft:
        new_line = resize(row.T, ratio).T
        stack = np.vstack((stack, new_line))
    stack = stack[1:stack.shape[0], :]
    stack = ex2_helper.phase_vocoder(stack, ratio)
    return ex2_helper.istft(stack).astype(data.dtype)


def conv_der(im):
    dx = scipy.signal.convolve2d(im, np.array([[-.5, 0, .5]]), "same")
    dy = scipy.signal.convolve2d(im, np.array([[-.5, 0, .5]]).reshape((3, 1)),
                                 "same")
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def fourier_der(im):
    im_dft = DFT2(im)
    shifted = np.fft.fftshift(im_dft)
    n, m = im.shape
    u_temp = np.arange(n) - n // 2
    u_final = u_temp * 2 * np.pi * 1j / n
    F_term_u = (shifted.T * u_final).T
    unshifted_u = np.fft.ifftshift(F_term_u)
    F_rev_u = np.real(IDFT2(unshifted_u))

    v_temp = np.arange(m) - m // 2
    v_final = v_temp * 2 * np.pi * 1j / m
    F_term_v = shifted * v_final
    unshifted_v = np.fft.ifftshift(F_term_v)
    F_rev_v = np.real(IDFT2(unshifted_v))

    return np.sqrt(F_rev_u ** 2 + F_rev_v ** 2)


if __name__ == "__main__":
    # change_rate("./ex2_presubmit/external/aria_4kHz.wav", 0.5)
    # change_samples("./ex2_presubmit/external/aria_4kHz.wav", 0.5)

    read = _read_file("./ex2_presubmit/external/aria_4kHz.wav")
    _save_file("test2.wav", resize_vocoder(read[1], 2), read[0])

    img = np.asarray(imageio.imread("./ex2_presubmit/external/monkey.jpg"),
                     dtype=np.float64) / 255
    img = skimage.color.rgb2gray(img)

    # plt.imshow(img)
    # plt.show()
    # plt.imshow(conv_der(img))
    # plt.show()
    plt.imshow(fourier_der(img))
    plt.show()
    print("done")
