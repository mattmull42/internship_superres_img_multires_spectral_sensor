import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, structural_similarity


def mse_errors(Ux, Ux_hat, bgr_channels=None):
    if bgr_channels:
        return [mean_squared_error(Ux[:, :, bgr_channels[i]], Ux_hat[:, :, i]) for i in range(3)]

    return [mean_squared_error(Ux[:, :, i], Ux_hat[:, :, i]) for i in range(3)]


def ssim_errors(Ux, Ux_hat, bgr_channels=None):
    if bgr_channels:
        return [structural_similarity(Ux[:, :, bgr_channels[i]], Ux_hat[:, :, i]) for i in range(3)]

    return [structural_similarity(Ux[:, :, i], Ux_hat[:, :, i]) for i in range(3)]


def get_indices_rgb(spectral_stencil):
    return (np.abs(spectral_stencil - 4450)).argmin(), (np.abs(spectral_stencil - 5500)).argmin(), (np.abs(spectral_stencil - 6500)).argmin()


def plot_images(images_list, titles_list, nrows=1, ncols=None, save_to=None, police_color=None, rgb_channels=None):
    n = len(images_list)

    if not ncols:
        ncols = n

    if not police_color:
        police_color = 'white'
    
    plt.figure(figsize=(40, 40))

    for i in range(n):
        plt.subplot(nrows, ncols, i + 1)

        if len(images_list[i].shape) == 3 and images_list[i].shape[2] == 3:
            plt.imshow(np.flip(images_list[i], axis=2))
        
        elif len(images_list[i].shape) == 2 or (len(images_list[i].shape) == 3 and images_list[i].shape[2] == 1):
            plt.imshow(images_list[i], cmap='gray')

        else:
            if rgb_channels:
                plt.imshow(images_list[i][:, :, rgb_channels])

            else:
                plt.imshow(np.mean(images_list[i], axis=2), cmap='gray')
        
        plt.title(titles_list[i], color=police_color, fontsize=32)
        plt.xticks(color=police_color, fontsize=24)
        plt.yticks(color=police_color, fontsize=24)

    if save_to:
        plt.savefig(save_to)

    plt.show()


def plot_errors(errors_list, title, save_to=None, police_color=None):

    if not police_color:
        police_color = 'white'

    plt.figure(figsize=(20, 10))

    x = np.arange(3)
    plt.bar(x-0.2, errors_list[0], width=0.2, color='cyan')
    plt.bar(x, errors_list[1], width=0.2, color='orange')
    plt.bar(x+0.2, errors_list[2], width=0.2, color='green')
    plt.xticks(x, ['Blue channel MSEs', 'Green channel MSEs', 'Red channel MSEs'], color=police_color, fontsize=18)
    plt.yticks(color=police_color, fontsize=16)
    plt.ylabel('MSEs', color=police_color, fontsize=16)
    plt.grid(axis='y')
    plt.legend(['Bayer', 'Quad-Bayer', 'Quad-Bayer,\nbinning &\nsub-sampling'], fontsize=12)
    plt.title(title, color=police_color, fontsize=24)

    if save_to:
        plt.savefig(save_to)

    plt.show()