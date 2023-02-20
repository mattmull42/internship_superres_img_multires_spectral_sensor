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