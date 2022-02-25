import os
from enum import IntEnum
from os.path import join, exists
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import skimage.color

import sol1

#########################
# --- Kali fix (v2) --- #
#########################

#      TODO: Leave as is or change to your image directory     #
PATH = "./test/"
#      TODO: Leave as is or change to your image directory     #


class Representation(IntEnum):
    grayscale = 1
    rgb = 2


def test_hist_eq(path: str, filenames: List[str]):
    print("Testing histogram equalization...")
    for rep in list(Representation):
        for file in filenames:
            im = sol1.read_image(os.path.join(path, file), rep.value)
            print(f"equalizing {file} (as {rep.name})...")
            im_eq, hist, hist_eq = sol1.histogram_equalize(im)
            im_eq = np.clip(im_eq, 0, 1)
            cmap = "viridis" if rep == Representation.rgb else "gray"
            plt.imsave(
                f"{path}_equalized/{file.replace('.jpg', '')}_{rep.name}_eq.jpg", im_eq, cmap=cmap
            )


def check_mono_dec(errors, n_quant):
    for i in range(1, len(errors)):
        if errors[i] > errors[i - 1]:
            print(
                f"errors list is not monotonically decreasing at indices {i - 1}, {i} for n_quant={n_quant}"
            )
            plt.plot(errors)
            plt.show()
            # raise Exception("Test failed!")
            break


QUANTIZATION_SEGMENTS = [2, 3, 5, 10]


def test_quantization(path: str, filenames: List[str]):
    print("\nTesting quantization...")
    for rep in list(Representation):
        for file in filenames:
            im = sol1.read_image(os.path.join(path, file), rep.value)
            print(f"quanitzing {file} (as {rep.name})...")
            for n_quant in QUANTIZATION_SEGMENTS:
                im_quant, errors = sol1.quantize(im, n_quant, 100)
                im_quant = np.clip(im_quant, 0, 1)
                cmap = "viridis" if rep == Representation.rgb else "gray"
                plt.imsave(
                    f"{path}_quantized/{file.replace('.jpg', '')}_{rep.name}_{n_quant}_quant.jpg",
                    im_quant,
                    cmap=cmap,
                )
                check_mono_dec(errors, n_quant)


def test_rgb2yiq(path: str, filenames: List[str]):
    print("\nTesting RGB to YIQ...")
    for file in filenames:
        im = sol1.read_image(os.path.join(path, file), 2)
        np.testing.assert_almost_equal(sol1.rgb2yiq(im), skimage.color.rgb2yiq(im), decimal=3)


def test_yiq2rgb(path: str, filenames: List[str]):
    print("\nTesting YIQ to RGB...")
    for file in filenames:
        im = sol1.read_image(os.path.join(path, file), 2)
        np.testing.assert_almost_equal(sol1.yiq2rgb(im), skimage.color.yiq2rgb(im), decimal=3)


def create_test_dirs(path: str):
    if not os.path.exists(path):
        raise IOError(f"Illegal path name: {path}")

    subs = [join(path, "_equalized"), join(path, "_quantized")]
    for sub in subs:
        if not exists(sub):
            os.mkdir(sub)


def end_print():
    print("\n")
    print("------- Test ended -------")
    print("If any plt graphs poped, the test failed (non-monotonic decreasing series)")


def run_all():
    path = PATH
    create_test_dirs(path)
    filenames = list(filter(lambda name: name.endswith(".jpg"), os.listdir(path)))

    # You can comment these lines to test just part of your code.
    test_rgb2yiq(path, filenames)
    test_yiq2rgb(path, filenames)
    test_hist_eq(path, filenames)
    test_quantization(path, filenames)
    end_print()


if __name__ == "__main__":
    run_all()
