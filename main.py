#!/usr/bin/env python3

from os import listdir

from testing.adjoint_tests import *
from testing.batch_tests import *
from testing.circulant_test import *
from testing.pipeline_tests import *


INPUT_DIR = 'input/'
BATCH_ARRAY = [path.join('input_batch/', image_name) for image_name in listdir('input_batch/')]

CFA = 'quad_bayer'
BINNING = CFA == 'quad_bayera'
NOISE_LEVEL = 0
MAX_ITER = 500
EPS = 0.001
REAL_PIC = False

# run_adjoint_tests()

# run_circulant_tests()

# run_pipeline_tests([INPUT_DIR + '01690.png'], NOISE_LEVEL, pipeline_version=1)
# run_pipeline_tests([INPUT_DIR + '01690.png'], NOISE_LEVEL, pipeline_version=2, pipeline_parameters=[MAX_ITER, EPS])

# run_batch_tests(BATCH_ARRAY, NOISE_LEVEL, pipeline_version=1)
# run_batch_tests(BATCH_ARRAY, NOISE_LEVEL, pipeline_version=2, pipeline_parameters=[MAX_ITER, EPS])

input_name = '01690'
x, spectral_stencil = initialize_input(INPUT_DIR + input_name + '.png')

if REAL_PIC:
    input_name = 'ProxOnyx_0'
    x = initialize_inverse_input(INPUT_DIR + input_name + '.png')
    input_size = np.append(x.shape, 3)

else:
    input_size = x.shape

forward_op = forward_operator(CFA, BINNING, NOISE_LEVEL, input_size, spectral_stencil)
baseline_op = Inverse_problem(CFA, BINNING, NOISE_LEVEL, input_size, spectral_stencil)
admm_op = Inverse_problem_ADMM(CFA, BINNING, NOISE_LEVEL, input_size, spectral_stencil, MAX_ITER, EPS)

if REAL_PIC:
    acq = x

else:
    acq = forward_op(x)

res_baseline = baseline_op(acq)
res_admm = admm_op(acq)

print(f'MSE :  {mse_error(x, res_admm):.4f}')
print(f'SSIM : {ssim_error(x, res_admm):.4f}')

# plt.imshow(acq, cmap='gray')
# plt.show()
# plt.imshow(res_baseline)
# plt.show()
plt.imshow(res_admm)
plt.show()