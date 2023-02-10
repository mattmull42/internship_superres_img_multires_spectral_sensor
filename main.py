#!/usr/bin/env python3

import matplotlib.pyplot as plt

from src.forward_operator.adjoint_tests import *
from src.input_initialization import *
from src.inversions.baseline_method.inversion_baseline import *
from src.inversions.pyproximal_implementation.admm_pyprox import *
from src.inversions.pyproximal_implementation.fista_pyprox import *


INPUT_DIR = 'input/'

CFA = 'sparse_3'
BINNING = CFA == 'quad_bayer'
NOISE_LEVEL = 0
MAX_ITER = 300
EPS = 0.01
REAL_PIC = False

# run_adjoint_tests()

input_name = '01690'
x, spectral_stencil = initialize_input(INPUT_DIR + input_name + '.png')

if REAL_PIC:
    input_name = 'cc'
    x = initialize_inverse_input(INPUT_DIR + input_name + '.tiff')
    input_size = np.append(x.shape, 3)

else:
    input_size = x.shape

forward_op = forward_operator(CFA, BINNING, NOISE_LEVEL, input_size, spectral_stencil)
baseline_op = Inverse_problem(CFA, BINNING, NOISE_LEVEL, input_size, spectral_stencil)
admm_op = Inverse_problem_ADMM(CFA, BINNING, NOISE_LEVEL, input_size, spectral_stencil, MAX_ITER, EPS)
fista_op = Inverse_problem_FISTA(CFA, BINNING, NOISE_LEVEL, input_size, spectral_stencil, MAX_ITER, EPS)

if REAL_PIC:
    acq = x

else:
    acq = forward_op(x)

for cfa in ['bayer', 'sparse_3', 'quad_bayer']:
    forward_op = forward_operator(cfa, BINNING, NOISE_LEVEL, input_size, spectral_stencil)

    for _ in range(1000):
        x = np.random.random(input_size)
        y = np.random.random(forward_op.output_size).reshape(-1, order="F")

        A = forward_op.get_matrix()
        acq = forward_op(x)
        forward_mat_vec = A @ x.reshape(-1, order="F")
        f = np.linalg.norm(acq.reshape(-1, order="F") - forward_mat_vec)
        a = np.linalg.norm(forward_op.adjoint(acq).reshape(-1, order="F") - A.T @ forward_mat_vec)
        x = x.reshape(-1, order="F")
        t = np.abs(y @ A @ x - A.T @ y @ x)

        if a or f or t:
            print(f'{cfa}:')
            print(f'Forward: {f}')
            print(f'Adjoint: {a}')
            print(f'Test: {t}')

# res_baseline = baseline_op(acq)
# res_fista = fista_op(acq)
# res_admm = admm_op(acq)

# fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
# ax[0][0].imshow(x)
# ax[0][0].set_title('Reference')
# ax[0][1].imshow(acq, cmap='gray')
# ax[0][1].set_title(f'Raw image')
# ax[1][0].imshow(res_baseline)
# ax[1][0].set_title(f'Baseline')
# ax[1][1].imshow(res_fista)
# ax[1][1].set_title(f'FISTA')
# ax[1][2].imshow(res_admm)
# ax[1][2].set_title(f'ADMM')
# plt.show()






# from src.norms import *
# from src.ista import ISTA
# from src.fista import FISTA
# from time import perf_counter
# import pyproximal as pyp


# class Forward_operator_pylops(pyl.LinearOperator):
#     def __init__(self, forward_operator):
#         self.forward_operator = forward_operator
#         self.shape = (np.prod(forward_operator.output_size), np.prod(forward_operator.input_size))
#         self.dtype = np.dtype(None)
#         self.explicit = False


#     def _matvec(self, x):
#         return self.forward_operator(x.reshape(self.forward_operator.input_size, order='F')).reshape(-1, order='F')


#     def _rmatvec(self, y):
#         return self.forward_operator.adjoint(y.reshape(self.forward_operator.output_size, order='F')).reshape(-1, order='F')

# norm_L2 = L2(A=forward_op, b=forward_op(x))
# norm_L1 = L1(sigma=EPS)

# x_0 = forward_op.adjoint(forward_op(x))

# # res_pyp_ista = pyp.optimization.primal.ProximalGradient(pyp.L2(Op=Forward_operator_pylops(forward_op), b=forward_op(x).reshape(-1, order='F')), pyp.L1(), x0=x_0.reshape(-1), epsg=0.001, niter=MAX_ITER).reshape(x.shape, order='F')
# # res_pyp_fista = pyp.optimization.primal.ProximalGradient(pyp.L2(Op=Forward_operator_pylops(forward_op), b=forward_op(x).reshape(-1, order='F')), pyp.L1(), x0=x_0.reshape(-1), epsg=0.001, niter=MAX_ITER, acceleration='fista').reshape(x.shape, order='F')
# # print(norm_L2(res_pyp_ista) + norm_L1(res_pyp_ista), norm_L2(res_pyp_ista), norm_L1(res_pyp_ista))
# # print(norm_L2(res_pyp_fista) + norm_L1(res_pyp_fista), norm_L2(res_pyp_fista), norm_L1(res_pyp_fista))

# x_ista, hist_ista, final_iter_ista, time_ista = ISTA(MAX_ITER, norm_L2, norm_L1, x_0, verbose=True)
# x_fista, hist_fista, final_iter_fista, time_fista = FISTA(MAX_ITER, norm_L2, norm_L1, x_0, verbose=True)

# fig, ax = plt.subplots()
# ax.plot(hist_ista, label=f'ISTA  | {final_iter_ista} iteration | {time_ista:.2f}s | {hist_ista[-1]:.4f}')
# ax.plot(hist_fista, label=f'FISTA | {final_iter_fista} iteration | {time_fista:.2f}s | {hist_fista[-1]:.4f}')
# ax.legend()
# plt.show()