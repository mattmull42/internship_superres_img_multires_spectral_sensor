import numpy as np
from os import path, mkdir
import csv


def get_indices_rgb(spectral_stencil):
    if len(spectral_stencil) < 3:
        raise Exception('The image must have at least 3 channels, a red, a green and a blue.')
        
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

    if not path.isfile(path.join('output', 'errors_log.csv')):
        with open(path.join('output', 'errors_log.csv'), 'w') as errors_log:
            csvwriter = csv.writer(errors_log)
            csvwriter.writerow(['Date', 'Pipeline', 'Name', 'Elapsed time (s)', 'MSE mean', 'SSIM mean', 'MSE red error', 'MSE green error', 'MSE blue error', 'SSIM red error', 'SSIM green error', 'SSIM blue error'])

    if not path.isdir(path.join('output', 'forward_model_outputs')):
        mkdir(path.join('output', 'forward_model_outputs'))

    if not path.isdir(path.join('output', 'inverse_problem_outputs')):
        mkdir(path.join('output', 'inverse_problem_outputs'))

    if not path.isdir(path.join('output', 'inverse_problem_ADMM_outputs')):
        mkdir(path.join('output', 'inverse_problem_ADMM_outputs'))

    if not path.isdir(path.join('output', 'errors_outputs')):
        mkdir(path.join('output', 'errors_outputs'))

    if not path.isdir(path.join('output', 'forward_model_outputs', 'output_cfa')):
        mkdir(path.join('output', 'forward_model_outputs', 'output_cfa'))

    if not path.isdir(path.join('output', 'forward_model_outputs', 'output_binning')):
        mkdir(path.join('output', 'forward_model_outputs', 'output_binning'))

    if not path.isdir(path.join('output', 'inverse_problem_outputs', 'output_unbinned')):
        mkdir(path.join('output', 'inverse_problem_outputs', 'output_unbinned'))

    if not path.isdir(path.join('output', 'inverse_problem_outputs', 'output_demosaicing')):
        mkdir(path.join('output', 'inverse_problem_outputs', 'output_demosaicing'))


def reduce_dimensions(data):
    shape = data.shape

    if len(shape) == 3:
        res = np.zeros((shape[0] * shape[1], shape[2]))

        for k in range(shape[2]):
            res[:, k] = data[:, :, k].flatten('F')

    elif len(shape) == 4:
        res = np.zeros((shape[0] * shape[1], shape[2], shape[3]))

        for l in range(shape[3]):
            for k in range(shape[2]):
                res[:, k, l] = data[:, :, k, l].flatten('F')

    else:
        res = data.flatten('F')

    return res


def increase_dimensions(data, new_shape):
    return np.reshape(data, new_shape, order='F')