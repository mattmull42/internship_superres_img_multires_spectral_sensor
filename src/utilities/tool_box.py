import numpy as np
from os import path, mkdir


def get_indices_rgb(spectral_stencil):
    return (np.abs(spectral_stencil - 6500)).argmin(), (np.abs(spectral_stencil - 5500)).argmin(), (np.abs(spectral_stencil - 4450)).argmin()


def same_size_images(image_1, image_2, ignore_depth=False):
    if ignore_depth:
        return (len(image_1.shape) == len(image_2.shape) == 3) and (image_1.shape[0] == image_2.shape[0]) and (image_1.shape[1] == image_2.shape[1])

    return image_1.shape == image_2.shape


def check_init_parameters(cfa=None, binning=None):
    if cfa and cfa not in ['bayer', 'quad_bayer']:
            raise Exception('cfa must be bayer, quad_bayer.')
    
    if binning and not isinstance(binning, bool):
        raise Exception('binning must be a boolean.')


def create_output_dirs():
    if not path.isdir('output'):
        mkdir('output')

    if not path.isdir(path.join('output', 'forward_model_outputs')):
        mkdir(path.join('output', 'forward_model_outputs'))

    if not path.isdir(path.join('output', 'inverse_problem_outputs')):
        mkdir(path.join('output', 'inverse_problem_outputs'))

    if not path.isdir(path.join('output', 'errors_outputs')):
        mkdir(path.join('output', 'errors_outputs'))

    if not path.isdir(path.join('output', 'forward_model_outputs', 'output_cfa')):
        mkdir(path.join('output', 'forward_model_outputs', 'output_cfa'))

    if not path.isdir(path.join('output', 'forward_model_outputs', 'output_binning')):
        mkdir(path.join('output', 'forward_model_outputs', 'output_binning'))

    if not path.isdir(path.join('output', 'forward_model_outputs', 'output_subsampling')):
        mkdir(path.join('output', 'forward_model_outputs', 'output_subsampling'))

    if not path.isdir(path.join('output', 'inverse_problem_outputs', 'output_upscaling')):
        mkdir(path.join('output', 'inverse_problem_outputs', 'output_upscaling'))

    if not path.isdir(path.join('output', 'inverse_problem_outputs', 'output_demosaicing')):
        mkdir(path.join('output', 'inverse_problem_outputs', 'output_demosaicing'))


def reduce_dimensions(data):
    shape = data.shape

    if len(shape) == 3:
        res = np.zeros((shape[0] * shape[1], shape[2]))

        for k in range(shape[2]):
            res[:, k] = data[:, :, k].flatten('F')

    else:
        res = data.flatten('F')

    return res


def increase_dimensions(data, new_shape):
    return np.reshape(data, new_shape, order='F')