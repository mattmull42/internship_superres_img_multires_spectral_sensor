from os import path
from PIL import Image
from netCDF4 import Dataset
import numpy as np


def initialize_input(image_path):
    if image_path.endswith('.ns'):
        dataset = Dataset(image_path, 'r')
        dataset.set_auto_mask(False)

        res = dataset.groups['radiance group'].variables['radiance matrix'][:, :, :].astype(float)
        spectral_stencil = dataset.variables['spectral stencil'][:]

    elif image_path == 'random':
        N_i, N_j = 16, 16
        res = np.zeros((N_i, N_j, 3))
        spectral_stencil = np.array([6500, 5500, 4450])

        for i in range(N_i):
            for j in range(N_j):
                res[i, j] = [np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]

    else:
        res = np.asarray(Image.open(image_path)) / 255
        spectral_stencil = np.array([6500, 5500, 4450])

        if res.shape[2] > 3:
            return res[:, :, :3], spectral_stencil

    return res, spectral_stencil


def initialize_inverse_input(image_path):
    res = np.asarray(Image.open(image_path)) / 255

    if len(res.shape) > 2:
        return res[:, :, 0]

    return res