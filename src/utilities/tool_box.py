import numpy as np
import torch
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


def get_inverse_input_path(forward_input_path, cfa, binning, noise_level):
    input_name_without_extension = path.basename(path.splitext(forward_input_path)[0])

    if not binning:
        return path.join('output', 'forward_model_outputs', input_name_without_extension + f'_noise_{noise_level}_{cfa}.png')

    return path.join('output', 'forward_model_outputs', input_name_without_extension + f'_noise_{noise_level}_binned_{cfa}.png')


def create_output_dirs():
    if not path.isdir('output'):
        mkdir('output')

    if not path.isfile(path.join('output', 'pipeline_log.csv')):
        with open(path.join('output', 'pipeline_log.csv'), 'w') as pipeline_log:
            csv_writer = csv.writer(pipeline_log)
            csv_writer.writerow(['Date', 'CFA', 'Binning', 'Noise level', 'Pipeline', 'Name', 'Elapsed time (s)', 'MSE', 'SSIM', 'MSE red', 'MSE green', 'MSE blue', 'SSIM red', 'SSIM green', 'SSIM blue'])

    if not path.isfile(path.join('output', 'batch_log.csv')):
        with open(path.join('output', 'batch_log.csv'), 'w') as batch_log:
            csv_writer = csv.writer(batch_log)
            csv_writer.writerow(['Date', 'CFA', 'Binning', 'Noise level', 'Pipeline', 'MSE', 'SSIM'])

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
        return np.reshape(data, (shape[0] * shape[1], shape[2]), order='F')

    elif len(shape) == 4:
        return np.reshape(data, (shape[0] * shape[1], shape[2], shape[3]), order='F')

    return np.reshape(data, (shape[0] * shape[1]), order='F')


def increase_dimensions(data, new_shape):
    return np.reshape(data, new_shape, order='F')


def reduce_dims_tensor(tensor):
    shape = tensor.shape

    if len(shape) == 2:
        return tensor.t().flatten()

    elif len(shape) == 3:
        return torch.transpose(tensor, 0, 1).flatten(end_dim=-2)


def increase_dims_tensor(tensor, shape):
    if len(shape) == 2:
        return torch.reshape(tensor, (shape[1], shape[0])).t()

    elif len(shape) == 3:
        return torch.transpose(torch.reshape(tensor, (shape[2], shape[1], shape[0])), 0, 2)