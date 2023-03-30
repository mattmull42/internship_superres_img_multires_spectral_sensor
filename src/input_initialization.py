from PIL import Image
import numpy as np


def initialize_input(image_path):
    res = np.array(Image.open(image_path)) / 255
    spectral_stencil = np.array([480, 525, 650])

    if res.shape[2] > 3:
        return res[:, :, :3], spectral_stencil

    return res, spectral_stencil


def initialize_inverse_input(image_path):
    if image_path.endswith('.png'):
        res = np.array(Image.open(image_path)) / 255

    elif image_path.endswith('.tiff'):
        # res = np.array(Image.open(image_path)) / 4095
        res = np.array(Image.open(image_path), dtype='float')
        res /= np.max(res)

    if len(res.shape) > 2:
        return res[:, :, 0]

    return res