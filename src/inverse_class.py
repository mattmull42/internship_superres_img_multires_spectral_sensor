import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage
import cv2

from src.utilities.tool_box import *
from src.utilities.custom_convolution import *


class Inverse_problem:
    def __init__(self, cfa, binning, forward_model_parameters):
        check_init_parameters(cfa=cfa, binning=binning)

        self.cfa = cfa
        self.binning = binning
        self.cfa_mask = forward_model_parameters[0]
        self.noise_level = forward_model_parameters[1]

        if self.binning:
            self.output_size = np.append(forward_model_parameters[2], 3)
            self.binning_factor = forward_model_parameters[3]


    def __call__(self, image):
        self.input = image

        if len(self.input.shape) != 2:
            raise Exception('Input must be a 2 dimensional matrix.')

        if self.binning:
            self.apply_upscaling()
            self.output_sparse_channel = increase_dimensions(np.transpose(np.transpose(self.cfa_mask) * reduce_dimensions(self.output_upscaling)), self.output_size)

        else:
            self.output_size = np.append(self.input.shape, 3)

            if self.cfa in ['bayer', 'quad_bayer']:
                self.output_sparse_channel = increase_dimensions(np.transpose(np.transpose(self.cfa_mask) * reduce_dimensions(self.input)), self.output_size)

            elif self.cfa == 'sparse_3':
                self.output_sparse_channel = np.zeros(self.output_size)
                self.output_sparse_channel[:, :, 0] = self.input
                self.output_sparse_channel[:, :, 1] = self.input
                self.output_sparse_channel[:, :, 2] = self.input

                self.output_sparse_channel[::8, ::8, 1:3] = 0
                self.output_sparse_channel[::8, ::8, 0] = self.input[::8, ::8]

                self.output_sparse_channel[::8, 4::8, ::2] = 0
                self.output_sparse_channel[::8, 4::8, 1] = self.input[::8, 4::8]
                self.output_sparse_channel[4::8, ::8, ::2] = 0
                self.output_sparse_channel[4::8, ::8, 1] = self.input[4::8, ::8]

                self.output_sparse_channel[4::8, 4::8, 0:2] = 0
                self.output_sparse_channel[4::8, 4::8, 2] = self.input[4::8, 4::8]

        if self.cfa == 'bayer':
            self.apply_bayer_demosaicing()

        elif self.cfa == 'quad_bayer':
            self.apply_quad_demosaicing()

        elif self.cfa == 'sparse_3':
            self.apply_sparse_3_demosaicing()

        self.output = self.output_demosaicing

        return self.output


    def apply_bayer_demosaicing(self):
        self.output_demosaicing = np.zeros(self.output_size)

        self.output_demosaicing[:, :, 0] = signal.convolve2d(self.output_sparse_channel[:, :, 0], ker_bayer_red_blue, mode='same')
        self.output_demosaicing[:, :, 1] = signal.convolve2d(self.output_sparse_channel[:, :, 1], ker_bayer_green, mode='same')
        self.output_demosaicing[:, :, 2] = signal.convolve2d(self.output_sparse_channel[:, :, 2], ker_bayer_red_blue, mode='same')


    def apply_quad_demosaicing(self):
        self.output_demosaicing = np.zeros(self.output_size)

        self.output_demosaicing[:, :, 0] = varying_kernel_convolution(self.output_sparse_channel[:, :, 0], K_list_red)
        self.output_demosaicing[:, :, 1] = varying_kernel_convolution(self.output_sparse_channel[:, :, 1], K_list_green)
        self.output_demosaicing[:, :, 2] = varying_kernel_convolution(self.output_sparse_channel[:, :, 2], K_list_blue)


    def apply_sparse_3_demosaicing(self):
        RGB_LR = self.output_sparse_channel[::4, ::4]
        RGB_LR[:, :, 0] = signal.convolve2d(RGB_LR[:, :, 0], ker_bayer_red_blue, mode='same')
        RGB_LR[:, :, 1] = signal.convolve2d(RGB_LR[:, :, 1], ker_bayer_green, mode='same')
        RGB_LR[:, :, 2] = signal.convolve2d(RGB_LR[:, :, 2], ker_bayer_red_blue, mode='same')

        W_HR = self.input.copy()

        for i in range(0, self.output_size[0], 4):
            for j in range(0, self.output_size[1], 4):
                if i == 0:
                    if j == 0:
                        W_HR[0, 0] = (W_HR[0, 1] + W_HR[1, 0] + W_HR[1, 1]) / 3

                    elif (j == 4 * (self.output_size[1] // 4)) and (j == self.output_size[1] - 1):
                        W_HR[0, j] = (W_HR[0, j - 1] + W_HR[1, j - 1] + W_HR[1, j]) / 3

                    else:
                        W_HR[0, j] = (W_HR[0, j - 1] + W_HR[0, j + 1] + W_HR[1, j - 1] + W_HR[1, j] + W_HR[1, j + 1]) / 5

                elif (i == 4 * (self.output_size[0] // 4)) and (i == self.output_size[0] - 1):
                    if j == 0:
                        W_HR[i, 0] = (W_HR[i - 1, 0] + W_HR[i - 1, 1] + W_HR[i, 1]) / 3

                    elif (j == 4 * (self.output_size[1] // 4)) and (j == self.output_size[1] - 1):
                        W_HR[i, j] = (W_HR[i - 1, j - 1] + W_HR[i - 1, j] + W_HR[i, j - 1]) / 3

                    else:
                        W_HR[i, j] = (W_HR[i - 1, j - 1] + W_HR[i - 1, j] + W_HR[i - 1, j + 1] + W_HR[i, j - 1] + W_HR[i, j + 1]) / 5

                else:
                    if j == 0:
                        W_HR[i, 0] = (W_HR[i - 1, 0] + W_HR[i - 1, 1] + W_HR[i, j + 1] + W_HR[i - 1, 0] + W_HR[i + 1, j + 1]) / 5

                    elif (j == 4 * (self.output_size[1] // 4)) and (j == self.output_size[1] - 1):
                        W_HR[i, j] = (W_HR[i - 1, j - 1] + W_HR[i - 1, j] + W_HR[i, j - 1] + W_HR[i + 1, j - 1] + W_HR[i + 1, j]) / 5

                    else:
                        W_HR[i, j] = (W_HR[i - 1, j - 1] + W_HR[i - 1, j] + W_HR[i - 1, j + 1] + W_HR[i, j - 1] + W_HR[i, j + 1] + W_HR[i + 1, j - 1] + W_HR[i + 1, j] + W_HR[i + 1, j + 1]) / 8

        RGB_LF_HR = np.repeat(np.repeat(RGB_LR, 4, axis=0), 4, axis=1)

        if self.output_size[0] % 2:
            RGB_LF_HR = RGB_LF_HR[:-1]

        if self.output_size[1] % 2:
            RGB_LF_HR = RGB_LF_HR[:, :-1]

        Y_LF_HR = np.average(RGB_LF_HR, axis=2, weights=[0.2126, 0.7152, 0.0722])

        self.output_demosaicing = np.zeros(self.output_size)

        self.output_demosaicing[:, :, 0] = RGB_LF_HR[:, :, 0] + W_HR[:, :] - Y_LF_HR[:, :]
        self.output_demosaicing[:, :, 1] = RGB_LF_HR[:, :, 1] + W_HR[:, :] - Y_LF_HR[:, :]
        self.output_demosaicing[:, :, 2] = RGB_LF_HR[:, :, 2] + W_HR[:, :] - Y_LF_HR[:, :]

        np.clip(self.output_demosaicing, 0, 1, self.output_demosaicing)


    def apply_upscaling(self):
        self.output_upscaling = np.repeat(np.repeat(self.input, 2, axis=0), 2, axis=1)

        if self.output_size[0] % 2:
            self.output_upscaling = self.output_upscaling[:-1]

        if self.output_size[1] % 2:
            self.output_upscaling = self.output_upscaling[:, :-1]


    def save_output(self, input_name):
        create_output_dirs()
        output_dir = path.join('output', 'inverse_problem_outputs')
        input_name_without_extension = path.basename(path.splitext(input_name)[0])

        if not self.binning:
            plt.imsave(path.join(output_dir, 'output_demosaicing', input_name_without_extension + '_demosaiced.png'), self.output_demosaicing)
            plt.imsave(path.join(output_dir, input_name_without_extension + '_reconstructed.png'), self.output)

        else:
            plt.imsave(path.join(output_dir, 'output_unbinned', input_name_without_extension + '_unbinned.png'), self.output_upscaling, cmap='gray')
            plt.imsave(path.join(output_dir, 'output_demosaicing', input_name_without_extension + '_demosaiced.png'), self.output_demosaicing)
            plt.imsave(path.join(output_dir, input_name_without_extension + '_reconstructed.png'), self.output)