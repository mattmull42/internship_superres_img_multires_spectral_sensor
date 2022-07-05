import matplotlib.pyplot as plt

from src.utilities.tool_box import *


def plot_images(images_list, titles_list, nrows=1, ncols=None, figsize=(40, 40), name=None, police_color='black', rgb_channels=None):
    n = len(images_list)

    if not ncols:
        ncols = n
    
    plt.figure(figsize=figsize)

    for i in range(n):
        plt.subplot(nrows, ncols, i + 1)

        if len(images_list[i].shape) == 3 and images_list[i].shape[2] == 3:
            plt.imshow(images_list[i])
        
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

    if name:
        create_output_dirs()
        plt.savefig(path.join('output', name))

    plt.show()


def plot_errors(errors_list, title, figsize=(16, 8), name=None, police_color='black'):
    plt.figure(figsize=figsize)

    x = np.arange(3)
    plt.bar(x-0.2, errors_list[0], width=0.2, color='cyan')
    plt.bar(x, errors_list[1], width=0.2, color='orange')
    plt.bar(x+0.2, errors_list[2], width=0.2, color='green')
    plt.xticks(x, ['Red channel', 'Green channel', 'Blue channel'], color=police_color, fontsize=18)
    plt.yticks(color=police_color, fontsize=16)
    plt.ylabel('Errors', color=police_color, fontsize=16)
    plt.grid(axis='y')
    plt.legend(['Bayer', 'Quad-Bayer', 'Quad-Bayer,\nbinning &\nsub-sampling'], fontsize=12)
    plt.title(title, color=police_color, fontsize=24)

    if name:
        create_output_dirs()
        plt.savefig(path.join('output', name))

    plt.show()