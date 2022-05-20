import matplotlib.pyplot as plt

from src.utilities.tool_box import *
from src.forward_operator_class import *
from src.operators.TV_operator_class import *


class Inverse_problem_ADMM:
    def __init__(self, cfa, binning, output_size, spectral_stencil, niter, sigma, eps, f=None):
        check_init_parameters(cfa=cfa, binning=binning)

        self.cfa = cfa
        self.binning = binning
        self.output_size = output_size
        self.spectral_stencil = spectral_stencil

        self.A = Forward_operator(self.cfa, self.output_size, self.spectral_stencil, self.binning)
        self.TV = TV_operator(self.output_size)
        self.L = odl.BroadcastOperator(self.A, self.TV)

        self.sigma = sigma
        self.eps = eps

        if f is not None:
            self.f = f
            self.op_norm = 1.1 * odl.power_method_opnorm(self.L, maxiter=50)
            self.tau = self.sigma / self.op_norm ** 2

        else:
            self.f = odl.solvers.ZeroFunctional(self.L.domain)
            self.tau = 1

        self.niter = niter


    def __call__(self, y, x=None):
        if x is not None:
            self.output = x

        else:
            if self.binning:
                self.output = self.A.adjoint(y) * self.A.binning_factor**2

            else:
                self.output = self.A.adjoint(y)

        self.data_fit = odl.solvers.L2NormSquared(self.A.range).translated(y)
        self.reg_func = self.eps * odl.solvers.GroupL1Norm(odl.ProductSpace(odl.rn(self.output_size[0] * self.output_size[1]), 2 * self.output_size[2]))
        self.g = odl.solvers.SeparableSum(self.data_fit, self.reg_func)

        odl.solvers.admm_linearized(self.output, self.f, self.g, self.L, self.tau, self.sigma, self.niter)

        self.output = self.output.asarray()

        np.clip(self.output, 0, 1, self.output)

        return self.output


    def save_output(self, input_name):
        create_output_dirs()
        output_dir = path.join('output', 'inverse_problem_ADMM_outputs')
        input_name_without_extension = path.basename(path.splitext(input_name)[0])

        plt.imsave(path.join(output_dir, input_name_without_extension + '_reconstructed_ADMM.png'), self.output)