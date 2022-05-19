import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from src.utilities.tool_box import *
from src.utilities.custom_convolution import *


class Inverse_problem:
    def __init__(self, cfa, binning, forward_model_parameters=None):
        check_init_parameters(cfa=cfa, binning=binning)

        self.cfa = cfa
        self.binning = binning

        if self.binning:
            self.output_size = np.append(forward_model_parameters[0], 3)
            self.binning_factor = forward_model_parameters[1]


    def __call__(self, image):
        self.input = image

        if len(self.input.shape) != 2:
            raise Exception('Input must be a 2 dimensional matrix.')

        if not self.binning:
            self.output_size = np.append(self.input.shape, 3)

        if self.binning:
            self.apply_upscaling()

        if self.cfa == 'bayer':
            self.apply_bayer_sparse()
            self.apply_bayer_demosaicing()

        elif self.cfa == 'quad_bayer':
            self.apply_quad_sparse()
            self.apply_quad_demosaicing()

        self.output = self.output_demosaicing

        return self.output


    def apply_bayer_sparse(self):
        self.output_sparse_channel = np.zeros(self.output_size)

        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                if i % 2 == 0 and j % 2 == 1:
                    self.output_sparse_channel[i, j, 0] = self.input[i, j]

                elif (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                    self.output_sparse_channel[i, j, 1] = self.input[i, j]

                elif i % 2 == 1 and j % 2 == 0:
                    self.output_sparse_channel[i, j, 2] = self.input[i, j]


    def apply_bayer_demosaicing(self):
        self.output_demosaicing = np.zeros(self.output_size)

        self.output_demosaicing[:, :, 0] = convolve2d(self.output_sparse_channel[:, :, 0], ker_bayer_red_blue, mode='same')
        self.output_demosaicing[:, :, 1] = convolve2d(self.output_sparse_channel[:, :, 1], ker_bayer_green, mode='same')
        self.output_demosaicing[:, :, 2] = convolve2d(self.output_sparse_channel[:, :, 2], ker_bayer_red_blue, mode='same')


    def apply_quad_sparse(self):
        self.output_sparse_channel = np.zeros(self.output_size)

        if self.binning:
            base_image = self.output_upscaling

        else:
            base_image = self.input

        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                if i % 4 < 2 and j % 4 >= 2:
                    self.output_sparse_channel[i, j, 0] = base_image[i, j]

                elif (i % 4 < 2 and j % 4 < 2) or (i % 4 >= 2 and j % 4 >= 2):
                    self.output_sparse_channel[i, j, 1] = base_image[i, j]

                elif i % 4 >= 2 and j % 4 < 2:
                    self.output_sparse_channel[i, j, 2] = base_image[i, j]


    def apply_quad_demosaicing(self):
        self.output_demosaicing = np.zeros(self.output_size)

        self.output_demosaicing[:, :, 0] = varying_kernel_convolution(self.output_sparse_channel[:, :, 0], K_list_red)
        self.output_demosaicing[:, :, 1] = varying_kernel_convolution(self.output_sparse_channel[:, :, 1], K_list_green)
        self.output_demosaicing[:, :, 2] = varying_kernel_convolution(self.output_sparse_channel[:, :, 2], K_list_blue)


    def apply_upscaling(self):
        self.output_upscaling = np.zeros(self.output_size[:2])
        P_i, P_j = int(np.ceil(self.output_size[0] / self.binning_factor)), int(np.ceil(self.output_size[1] / self.binning_factor))

        for i in range(P_i):
            for j in range(P_j):
                self.output_upscaling[i * self.binning_factor:(i + 1) * self.binning_factor, j * self.binning_factor:(j + 1) * self.binning_factor] = self.input[i, j]


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