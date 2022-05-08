from src.forward_class import *
from src.inverse_class import *
from src.utilities.input_initialization import *
from src.utilities.errors import *


class Pipeline:
    def __init__(self, cfa, binning):
        check_init_parameters(cfa, binning)

        self.cfa = cfa
        self.binning = binning


    def run(self, input_name):
        self.input_name_forward = input_name
        self.image_forward, self.spectral_stencil = initialize_input(self.input_name_forward)

        self.forward_model = Forward_model(self.cfa, self.binning)
        self.forward_model.set_input(self.image_forward, self.spectral_stencil, self.input_name_forward)
        self.forward_model.run()
        self.forward_model.save_results()

        input_name_without_extension = path.basename(path.splitext(self.input_name_forward)[0])

        if self.cfa == 'bayer':
            self.input_name_inverse = input_name_without_extension + '_bayer.png'

        elif self.cfa == 'quad_bayer':
            self.input_name_inverse = input_name_without_extension + '_quad_bayer.png'

        if self.cfa == 'quad_bayer' and self.binning:
            self.input_name_inverse = input_name_without_extension + '_binned_quad_bayer.png'

        self.image_inverse = initialize_inverse_input(self.input_name_inverse)

        if self.binning:
            self.inverse_problem = Inverse_problem(self.cfa, self.binning, self.forward_model.get_parameters())

        else:
            self.inverse_problem = Inverse_problem(self.cfa, self.binning)

        self.inverse_problem.set_input(self.image_inverse, self.input_name_inverse)
        self.inverse_problem.run()
        self.inverse_problem.save_results()

        self.get_errors()

        self.save_results()


    def get_errors(self):
        self.mse_errors = self.get_mse_errors()
        self.ssim_errors = self.get_ssim_errors()
        self.image_abs_difference = self.get_image_abs_difference()


    def get_mse_errors(self):
        return mse_errors(self.image_forward, self.inverse_problem.output, rgb_channels=get_indices_rgb(self.spectral_stencil))


    def get_ssim_errors(self):
        return ssim_errors(self.image_forward, self.inverse_problem.output, rgb_channels=get_indices_rgb(self.spectral_stencil))


    def get_image_abs_difference(self):
        return image_abs_diff(self.image_forward, self.inverse_problem.output, rgb_channels=get_indices_rgb(self.spectral_stencil))
    

    def save_results(self):
        create_output_dirs()
        plt.imsave(path.join('output', 'errors_outputs', self.input_name_inverse[:-4] + '_absolute_difference.png'), self.image_abs_difference, cmap='gray')