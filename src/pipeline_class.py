from src.forward_class import *
from src.inverse_class import *
from src.utilities.input_initialization import *


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