from scipy import signal
from PIL import Image

from .custom_convolution import *
from src.forward_operator.operators.misc.cfa_masks import *


class Inverse_problem:
    def __init__(self, cfa, binning, noise_level, output_size, spectral_stencil):
        self.cfa = cfa
        self.binning = binning
        self.noise_level = noise_level
        self.output_size = output_size

        if self.cfa == 'bayer':
            self.cfa_mask = get_bayer_mask(self.output_size, spectral_stencil)

        elif self.cfa == 'quad_bayer':
            self.cfa_mask = get_quad_mask(self.output_size, spectral_stencil)

        elif self.cfa == 'sparse_3':
            self.cfa_mask = get_sparse_3_mask(self.output_size, spectral_stencil)


    def __call__(self, image):
        self.input = image

        if len(self.input.shape) != 2:
            raise Exception('Input must be a 2 dimensional matrix.')

        if self.binning:
            self.apply_upscaling()
            self.output_sparse_channel = self.cfa_mask * self.output_upscaling[..., np.newaxis]

        else:
            if self.cfa in ['bayer', 'quad_bayer']:
                self.output_sparse_channel = self.cfa_mask * self.input[..., np.newaxis]

            elif self.cfa == 'sparse_3':
                self.output_sparse_channel = np.zeros(self.output_size)
                self.output_sparse_channel[:, :, 0] = self.input
                self.output_sparse_channel[:, :, 1] = self.input
                self.output_sparse_channel[:, :, 2] = self.input

                self.output_sparse_channel[::8, ::8, 1:3] = 0

                self.output_sparse_channel[::8, 4::8, ::2] = 0
                self.output_sparse_channel[4::8, ::8, ::2] = 0

                self.output_sparse_channel[4::8, 4::8, 0:2] = 0

        if self.cfa == 'bayer':
            self.apply_bayer_demosaicing()

        elif self.cfa == 'quad_bayer':
            self.apply_quad_demosaicing()

        elif self.cfa == 'sparse_3':
            self.apply_sparse_3_demosaicing()

        self.output = self.output_demosaicing

        return self.output


    def apply_bayer_demosaicing(self):
        self.output_demosaicing = np.empty(self.output_size)

        self.output_demosaicing[:, :, 0] = signal.convolve2d(self.output_sparse_channel[:, :, 0], ker_bayer_red_blue, mode='same')
        self.output_demosaicing[:, :, 1] = signal.convolve2d(self.output_sparse_channel[:, :, 1], ker_bayer_green, mode='same')
        self.output_demosaicing[:, :, 2] = signal.convolve2d(self.output_sparse_channel[:, :, 2], ker_bayer_red_blue, mode='same')


    def apply_quad_demosaicing(self):
        self.output_demosaicing = np.empty(self.output_size)

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
                        W_HR[0, 0] = (2 * self.input[0, 1] + 2 * self.input[1, 0] + self.input[1, 1]) / 5

                    elif (j == 4 * (self.output_size[1] // 4)) and (j == self.output_size[1] - 1):
                        W_HR[0, j] = (2 * self.input[0, j - 1] + self.input[1, j - 1] + 2 * self.input[1, j]) / 5

                    else:
                        W_HR[0, j] = (2 * self.input[0, j - 1] + 2 * self.input[0, j + 1] + self.input[1, j - 1] + 2 * self.input[1, j] + self.input[1, j + 1]) / 8

                elif (i == 4 * (self.output_size[0] // 4)) and (i == self.output_size[0] - 1):
                    if j == 0:
                        W_HR[i, 0] = (2 * self.input[i - 1, 0] + self.input[i - 1, 1] + 2 * self.input[i, 1]) / 5

                    elif (j == 4 * (self.output_size[1] // 4)) and (j == self.output_size[1] - 1):
                        W_HR[i, j] = (self.input[i - 1, j - 1] + 2 * self.input[i - 1, j] + 2 * self.input[i, j - 1]) / 5

                    else:
                        W_HR[i, j] = (self.input[i - 1, j - 1] + 2 * self.input[i - 1, j] + self.input[i - 1, j + 1] + 2 * self.input[i, j - 1] + 2 * self.input[i, j + 1]) / 8

                else:
                    if j == 0:
                        W_HR[i, 0] = (2 * self.input[i - 1, 0] + self.input[i - 1, 1] + 2 * self.input[i, 1] + 2 * self.input[i - 1, 0] + self.input[i + 1, 1]) / 8

                    elif (j == 4 * (self.output_size[1] // 4)) and (j == self.output_size[1] - 1):
                        W_HR[i, j] = (self.input[i - 1, j - 1] + 2 * self.input[i - 1, j] + 2 * self.input[i, j - 1] + self.input[i + 1, j - 1] + 2 * self.input[i + 1, j]) / 8

                    else:
                        W_HR[i, j] = (self.input[i - 1, j - 1] + 2 * self.input[i - 1, j] + self.input[i - 1, j + 1] + 2 * self.input[i, j - 1] + 2 * self.input[i, j + 1] + self.input[i + 1, j - 1] + 2 * self.input[i + 1, j] + self.input[i + 1, j + 1]) / 12

        RGB_LF_HR = np.array(Image.fromarray((RGB_LR * 255).astype(np.uint8)).resize((4 * RGB_LR.shape[1], 4 * RGB_LR.shape[0]))) / 255

        if W_HR.shape[0] % 4:
            RGB_LF_HR = RGB_LF_HR[:-(4 - (W_HR.shape[0] % 4))]

        if W_HR.shape[1] % 4:
            RGB_LF_HR = RGB_LF_HR[:, :-(4 - (W_HR.shape[1] % 4))]

        Y_LF_HR = np.mean(RGB_LF_HR, axis=2)

        self.output_demosaicing = RGB_LF_HR + (W_HR - Y_LF_HR)[..., np.newaxis]

        np.clip(self.output_demosaicing, 0, 1, self.output_demosaicing)


    def apply_upscaling(self):
        if self.cfa == 'quad_bayer':
            self.output_upscaling = np.repeat(np.repeat(self.input, 2, axis=0), 2, axis=1)

            if self.output_size[0] % 2:
                self.output_upscaling = self.output_upscaling[:-1]

            if self.output_size[1] % 2:
                self.output_upscaling = self.output_upscaling[:, :-1]