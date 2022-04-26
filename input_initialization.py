from os import path
from PIL import Image
from netCDF4 import Dataset
import numpy as np


def initialize_input(kind):
    if kind == 'netCDF4':
        dataset = Dataset(path.join('data', 'multispectral_colorchecker.ns'), 'r')
        dataset.set_auto_mask(False)

        Ux = dataset.groups['radiance group'].variables['radiance matrix'][:, :, :].astype(float)
        spectral_stencil = dataset.variables['spectral stencil'][:]

    elif kind == 'random':
        Ux = np.zeros((10, 10, 3))
        spectral_stencil = np.array([4400, 5500, 6500])

        for i in range(10):
            for j in range(10):
                Ux[i, j] = [np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]

    else:
        Ux = np.flip(np.asarray(Image.open(path.join('data', kind + '.jpg')).resize((200, 200))), axis=2) / 255
        spectral_stencil = np.array([4400, 5500, 6500])

    return Ux, spectral_stencil