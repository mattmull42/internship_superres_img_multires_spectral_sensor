from skimage.metrics import mean_squared_error, structural_similarity

from src.utilities.tool_box import *


def mse_error(Ux, Ux_hat, rgb_channels=[0, 1, 2]):
    if not same_size_images(Ux, Ux_hat, ignore_depth=True):
        raise Exception('Both images must be of the same spatial size.')

    return mean_squared_error(Ux[:, :, rgb_channels], Ux_hat)


def ssim_error(Ux, Ux_hat, rgb_channels=[0, 1, 2]):
    if not same_size_images(Ux, Ux_hat, ignore_depth=True):
        raise Exception('Both images must be of the same spatial size.')

    return structural_similarity(Ux[:, :, rgb_channels], Ux_hat, channel_axis=2)


def image_abs_diff(Ux, Ux_hat, rgb_channels=[0, 1, 2]):
    if not same_size_images(Ux, Ux_hat, ignore_depth=True):
        raise Exception('Both images must be of the same spatial size.')

    return np.abs(Ux[:, :, rgb_channels] - Ux_hat)