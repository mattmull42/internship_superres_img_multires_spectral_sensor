import torch

from src.operators.cfa_operator_class import *
from src.operators.binning_operator_class import *


class Forward_operator(odl.Operator):
    def __init__(self, cfa, binning, noise_level, input_size, spectral_stencil):
        check_init_parameters(cfa=cfa, binning=binning)

        self.cfa = cfa
        self.binning = binning
        self.input_size = input_size
        self.noise_level = noise_level

        self.cfa_operator = cfa_operator(self.cfa, self.input_size, spectral_stencil)

        if self.binning:
            self.binning_operator = binning_operator(self.cfa, self.input_size[:2])
            self.binning_factor = self.binning_operator.l
            self.output_size = self.binning_operator.P_i, self.binning_operator.P_j

        else:
            self.output_size = self.input_size[:2]

        if self.binning:
            if self.cfa == 'quad_bayer':
                self.adjoint_op = odl.OperatorComp(self.cfa_operator.adjoint, self.binning_operator.adjoint)

        else:
            self.adjoint_op = self.cfa_operator.adjoint

        odl.Operator.__init__(self, odl.uniform_discr([0, 0, 0], self.input_size, self.input_size), odl.uniform_discr([0, 0], self.output_size, self.output_size))


    def _call(self, Ux):
        self.output = self.cfa_operator(Ux)
        
        if self.noise_level != 0:
            self.add_noise()

        if self.binning:
            if self.cfa == 'quad_bayer':
                self.output = self.binning_operator(self.output)

        return self.output


    @property
    def adjoint(self):
        return self.adjoint_op


    def add_noise(self):
        self.output = np.clip(self.output + np.random.normal(0, self.noise_level / 100, self.output.shape), 0, 1)


    def get_parameters(self):
        if self.binning:
            return self.cfa_operator.cfa_mask, self.noise_level, self.input_size[:2], self.binning_factor

        else:
            return self.cfa_operator.cfa_mask, self.noise_level


    def get_matrix_operator(self, is_tensor=False):
        if not hasattr(self, 'matrix_operator'):
            if self.binning:
                self.matrix_operator = (self.binning_operator.get_matrix_operator() @ self.cfa_operator.get_matrix_operator()).tocoo()

            else:
                self.matrix_operator = self.cfa_operator.get_matrix_operator()

        if is_tensor:
            return scipy_sparse_to_pytorch_sparse(self.matrix_operator)

        return self.matrix_operator


    def apply_matrix_operator(self, x, is_tensor=False, dtype=None):
        self.get_matrix_operator(is_tensor)

        if is_tensor:
            if self.noise_level != 0:
                tmp = scipy_sparse_to_pytorch_sparse(self.cfa_operator.matrix_operator, dtype) @ reduce_dims_tensor(reduce_dims_tensor(x))

                if dtype:
                    tmp = torch.clamp(tmp + torch.normal(0, self.noise_level / 100, [self.cfa_operator.matrix_operator.shape[0]]).type(dtype), 0, 1).type(dtype)

                else:
                    tmp = torch.clamp(tmp + torch.normal(0, self.noise_level / 100, [self.cfa_operator.matrix_operator.shape[0]]), 0, 1)

                if self.binning:
                    return increase_dims_tensor(scipy_sparse_to_pytorch_sparse(self.binning_operator.matrix_operator, dtype) @ tmp, self.output_size)

                return increase_dims_tensor(tmp, self.output_size)

            return increase_dims_tensor(scipy_sparse_to_pytorch_sparse(self.matrix_operator, dtype) @ reduce_dims_tensor(reduce_dims_tensor(x)), self.output_size)

        else:
            if self.noise_level != 0:
                tmp = self.cfa_operator.matrix_operator @ reduce_dimensions(reduce_dimensions(x))
                tmp = np.clip(tmp + np.random.normal(0, self.noise_level / 100, self.cfa_operator.matrix_operator.shape[0]), 0, 1)

                if self.binning:
                    return increase_dimensions(self.binning_operator.matrix_operator @ tmp, self.output_size)

                return increase_dimensions(tmp, self.output_size)

            return increase_dimensions(self.matrix_operator @ reduce_dimensions(reduce_dimensions(x)), self.output_size)


    def save_output(self, input_name):
        create_output_dirs()

        self.cfa_operator.save_output(input_name + f'_noise_{self.noise_level}')
        
        if self.binning:
            if self.cfa == 'quad_bayer':
                file_name = path.basename(path.splitext(input_name)[0])+ f'_noise_{self.noise_level}_binned_' + self.cfa + '.png'
                self.binning_operator.save_output(input_name + f'_noise_{self.noise_level}')

        else:
            file_name = path.basename(path.splitext(input_name)[0]) + f'_noise_{self.noise_level}_' + self.cfa + '.png'

        plt.imsave(path.join('output', 'forward_model_outputs', file_name), self.output, cmap='gray')