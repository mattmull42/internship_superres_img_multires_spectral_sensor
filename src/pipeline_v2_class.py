from os import path

from src.forward_operator_class import *
from src.inverse_admm_class import *
from src.utilities.input_initialization import *
from src.utilities.errors import *


class Pipeline_v2:
    def __init__(self, cfa, binning, niter, sigma, epsilon, box_constraint):
        check_init_parameters(cfa=cfa, binning=binning)

        self.cfa = cfa
        self.binning = binning
        self.niter = niter
        self.sigma = sigma
        self.epsilon = epsilon
        self.box_constraint = box_constraint


    def run(self, input_path, noise_level):
        self.input_path_forward = input_path
        self.image_forward, self.spectral_stencil = initialize_input(self.input_path_forward)
        self.input_size = self.image_forward.shape
        self.noise_level = noise_level

        self.forward_model = Forward_operator(self.cfa, self.input_size, self.spectral_stencil, self.binning, self.noise_level)
        self.forward_model(self.image_forward)

        self.inverse_problem = Inverse_problem_ADMM(self.cfa, self.binning, self.noise_level, self.input_size, self.spectral_stencil, self.niter, self.sigma, self.epsilon, self.box_constraint)
        self.inverse_problem(self.forward_model.output.asarray())

        self.get_errors()


    def get_errors(self):
        self.get_mse_errors()
        self.get_ssim_errors()
        self.get_image_abs_difference()


    def get_mse_errors(self):
        self.mse_errors = mse_errors(self.image_forward, self.inverse_problem.output, rgb_channels=get_indices_rgb(self.spectral_stencil))


    def get_ssim_errors(self):
        self.ssim_errors = ssim_errors(self.image_forward, self.inverse_problem.output, rgb_channels=get_indices_rgb(self.spectral_stencil))


    def get_image_abs_difference(self):
        self.image_abs_difference = image_abs_diff(self.image_forward, self.inverse_problem.output, rgb_channels=get_indices_rgb(self.spectral_stencil))


    def save_output(self):
        input_path_inverse = get_inverse_input_path(self.input_path_forward, self.cfa, self.binning, self.noise_level)

        self.forward_model.save_output(path.basename(self.input_path_forward))
        self.inverse_problem.save_output(path.basename(input_path_inverse))

        plt.imsave(path.join('output', 'errors_outputs', path.basename(input_path_inverse)[:-4] + f'_noise_{self.noise_level}_ADMM_absolute_difference.png'), self.image_abs_difference, cmap='gray')