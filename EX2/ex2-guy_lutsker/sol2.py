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


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect',
                                  order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


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
    :return:DFT of image
    """
    DFT_rows = np.apply_along_axis(DFT, 0, image)
    DFT_all = np.apply_along_axis(DFT, 1, DFT_rows)
    return DFT_all


def IDFT2(fourier_image):
    """

    :param fourier_image:a real image transformed with DFT2
    :return:inverse DFT of fourier_image
    """
    IDFT_rows = np.apply_along_axis(IDFT, 0, fourier_image)
    IDFT_all = np.apply_along_axis(IDFT, 1, IDFT_rows)
    return IDFT_all


def _read_file(filename):
    """
    reads wav file
    :param filename:
    :return:
    """
    return scipy.io.wavfile.read(filename)


def _save_file(filename, data, rate):
    """
    saves wav file
    :param filename:
    :param data:
    :param rate:
    :return:
    """
    scipy.io.wavfile.write(filename, rate, data)


def change_rate(filename, ratio):
    """
    changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header
    :param filename:a string representing the path to a WAV file,
    :param ratio: a positive float64 representing the duration change.
    :return:
    """
    rate, data = _read_file(filename)
    _save_file("change_rate.wav", data, int(rate * ratio))


def change_samples(filename, ratio):
    """
    fast forward function that changes the duration of an audio file by reducing the number of samples
    using Fourier
    :param filename:a string representing the path to a WAV file,
    :param ratio: a positive float64 representing the duration change.
    :return: deformed data
    """
    rate, data = _read_file(filename)
    data_new = resize(data, ratio).astype("float64")
    _save_file("change_samples.wav", data_new, rate)
    return data_new


def resize(data, ratio):
    """
    changes the number of samples by the given ratio
    :param data: data
    :param ratio: ratio
    :return:
    """
    data_dft = DFT(data)
    shifted = np.fft.fftshift(data_dft)
    n = len(shifted)
    if ratio > 1:
        cropped = shifted[int((n - int(n / ratio)) / 2):
                          -int(np.ceil((n - int(n / ratio)) / 2))]
        res = cropped
    else:
        # ratio < 1
        padding = int(np.floor(n * (1 / (2 * ratio)) - n / 2))
        flag = 0
        if padding * 2 + n < int(n / ratio):
            flag = 1
        padded = np.pad(shifted, (padding, padding + 1 * flag),
                        'constant', constant_values=(0, 0))
        res = padded
    res = np.fft.ifftshift(res)
    return IDFT(res)


def resize_spectrogram(data, ratio):
    """
    a function that speeds up a WAV file, without changing the pitch, using spectrogram scaling
    :param data: data
    :param ratio: ratio
    :return:
    """
    data_stft = stft(data)
    stack = np.apply_along_axis(resize, 1, data_stft, ratio)
    return istft(stack).astype(data.dtype)


def resize_vocoder(data, ratio):
    """
     speedups a WAV file by phase vocoding its spectrogram
    :param data:
    :param ratio:
    :return:
    """
    data_new = phase_vocoder(stft(data), ratio)
    return istft(data_new).astype(data.dtype)


def conv_der(im):
    """
    function that computes the magnitude of image derivatives using convolution
    :param im: image
    :return: image der
    """
    dx = scipy.signal.convolve2d(im, np.array([[-.5, 0, .5]]), "same")
    dy = scipy.signal.convolve2d(im, np.array([[-.5, 0, .5]]).reshape((3, 1)),
                                 "same")
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def fourier_der(im):
    """
    function that computes the magnitude of image derivatives using fourier
    :param im: image
    :return: image der
    """
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
    # t = DFT(np.arange(50))
    #
    # # change_rate("./ex2_presubmit/external/aria_4kHz.wav", 0.5)
    # change_samples("./ex2_presubmit/external/aria_4kHz.wav", 2)
    #
    # read = _read_file("./ex2_presubmit/external/aria_4kHz.wav")
    # _save_file("test2.wav", resize_vocoder(read[1], 2), read[0])

    img = np.asarray(imageio.imread("./ex2_presubmit/external/monkey.jpg"),
                     dtype=np.float64) / 255
    img = skimage.color.rgb2gray(img)

    # plt.imshow(img)
    # plt.show()
    # plt.imshow(conv_der(img))
    # plt.show()
    ttt = fourier_der(img)
    plt.imshow(ttt)
    plt.show()
    print("do ne")
    print("hello er      an")
