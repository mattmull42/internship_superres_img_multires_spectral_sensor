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
    return [np.linalg.norm((Ux[:, :, i] - Ux_hat[:, :, i]) / 255) / (Ux.shape[0] * Ux.shape[1]) for i in range(3)]