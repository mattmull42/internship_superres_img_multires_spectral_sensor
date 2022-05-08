from skimage.metrics import mean_squared_error, structural_similarity

from src.utilities.tool_box import *


def mse_errors(Ux, Ux_hat, rgb_channels=None):
    if not same_size_images(Ux, Ux_hat, ignore_depth=True):
        raise Exception('Both images must be of the same spatial size.')

    if rgb_channels:
        return [mean_squared_error(Ux[:, :, rgb_channels[i]], Ux_hat[:, :, i]) for i in range(3)]

    return [mean_squared_error(Ux[:, :, i], Ux_hat[:, :, i]) for i in range(3)]


def ssim_errors(Ux, Ux_hat, rgb_channels=None):
    if not same_size_images(Ux, Ux_hat, ignore_depth=True):
        raise Exception('Both images must be of the same spatial size.')

    if rgb_channels:
        return [structural_similarity(Ux[:, :, rgb_channels[i]], Ux_hat[:, :, i]) for i in range(3)]

    return [structural_similarity(Ux[:, :, i], Ux_hat[:, :, i]) for i in range(3)]


def image_abs_diff(Ux, Ux_hat, rgb_channels=None):
    if not same_size_images(Ux, Ux_hat, ignore_depth=True):
        raise Exception('Both images must be of the same spatial size.')

    res = np.zeros((Ux.shape[0], Ux.shape[1], 3))

    if rgb_channels:
        for i in range(3):
            res[:, :, i] = np.abs(Ux[:, :, rgb_channels[i]] - Ux_hat[:, :, i])

        return res

    for i in range(3):
        res[:, :, i] = np.abs(Ux[:, :, rgb_channels[i]] - Ux_hat[:, :, i])

    return res