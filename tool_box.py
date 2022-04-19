import numpy as np
import matplotlib.pyplot as plt


def plot_images(images_list, titles_list, nrows=1, ncols=None, save_to=None, police_color=None):
    n = len(images_list)

    if not ncols:
        ncols = n

    if not police_color:
        police_color = 'white'
    
    plt.figure(figsize=(40, 40))

    for i in range(n):
        if len(images_list[i].shape) == 3 and images_list[i].shape[2] == 3:
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(np.flip(images_list[i], axis=2))
        
        else:
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(images_list[i], cmap='gray')
        
        plt.title(titles_list[i], color=police_color, fontsize=32)
        plt.xticks(color=police_color, fontsize=24)
        plt.yticks(color=police_color, fontsize=24)

    if save_to:
        plt.savefig(save_to)

    plt.show()


def errors(Ux, Ux_hat):
    return [(((Ux[:, :, i] - Ux_hat[:, :, i]) / 255)**2).mean() for i in range(3)]


def plot_errors(errors_list, save_to=None, police_color=None):

    if not police_color:
        police_color = 'white'

    plt.figure(figsize=(20, 10))

    x = np.arange(3)
    plt.bar(x-0.2, errors_list[0], width=0.2, color='cyan')
    plt.bar(x, errors_list[1], width=0.2, color='orange')
    plt.bar(x+0.2, errors_list[2], width=0.2, color='green')
    plt.xticks(x, ['Blue channel MSEs', 'Green channel MSEs', 'Red channel MSEs'], color=police_color, fontsize=16)
    plt.yticks(color=police_color, fontsize=16)
    plt.ylabel("MSEs")
    plt.grid(axis='y')
    plt.legend(['Bayer', 'Quad-Bayer', 'Quad-Bayer,\nbinning &\nsub-sampling'], fontsize=10)
    plt.title('Mean Squares Errors for each channel for the 3 different studied options', color=police_color, fontsize=24)

    if save_to:
        plt.savefig(save_to)

    plt.show()