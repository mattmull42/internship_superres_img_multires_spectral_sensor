import jax.numpy as jnp
from scico import functional, linop, loss, metric
from scico.optimize import LinearizedADMM, PDHG
from scico.optimize.admm import ADMM, LinearSubproblemSolver

import matplotlib.pyplot as plt

from time import perf_counter

from src.input_initialization import *
from src.forward_operator.forward_operator import *
from src.inversions.baseline_method.inversion_baseline import *


#######################################
# General parameters
#######################################

INPUT_DIR = 'input/'

CFA = 'sparse_3'
BINNING = CFA == 'quad_bayer'
NOISE_LEVEL = 0
MAX_ITER = 100


#######################################
# Ground truth and raw acquisition
#######################################

input_name = '01690'
gt, spectral_stencil = initialize_input(INPUT_DIR + input_name + '.png')
gt = jnp.array(gt)
input_size = gt.shape

forward_op = forward_operator(CFA, BINNING, NOISE_LEVEL, input_size, spectral_stencil)

acq = jnp.array(forward_op(gt))


#######################################
# Data fidelity term
#######################################

def forward_pass(x):
    return jnp.array(forward_op(x))

def adjoint_pass(y):
    return jnp.array(forward_op.adjoint(y))

A = linop.LinearOperator(input_shape=input_size, output_shape=acq.shape, eval_fn=forward_pass, adj_fn=adjoint_pass)
f = loss.SquaredL2Loss(y=acq, A=A)


#######################################
# Baseline method
#######################################

baseline_inverse = Inverse_problem(CFA, BINNING, NOISE_LEVEL, input_size, spectral_stencil)
x_baseline = jnp.array(baseline_inverse(np.array(acq)))


#######################################
# TV ADMM
#######################################

eps = 1e-2

g = functional.L21Norm(l2_axis=(0, 3))
C = eps * linop.FiniteDifference(input_shape=gt.shape, append=0, axes=(0, 1))

rho = 2e1

solver_ADMM = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[rho],
    x0=adjoint_pass(acq),
    maxiter=MAX_ITER,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={'tol': 1e-3, 'maxiter': 100})
)


#######################################
# TV LADMM
#######################################

eps = 5e-3

g = functional.L21Norm(l2_axis=(0, 3))
C = eps * linop.FiniteDifference(input_shape=gt.shape, append=0, axes=(0, 1))
C_squared_norm = np.float64(linop.operator_norm(C))**2

nu = 1e0
mu = 0.8 * nu / C_squared_norm

solver_LADMM = LinearizedADMM(
    f=f,
    g=g,
    C=C,
    nu=nu,
    mu=mu,
    x0=adjoint_pass(acq),
    maxiter=MAX_ITER
)


#######################################
# TV PDHG
#######################################

eps = 5e-3

g = functional.L21Norm(l2_axis=(0, 3))
C = eps * linop.FiniteDifference(input_shape=gt.shape, append=0, axes=(0, 1))
C_squared_norm = np.float64(linop.operator_norm(C))**2

sigma = 1e2
tau = 0.99 / (sigma * C_squared_norm)

solver_PDHG = PDHG(
    f=f,
    g=g,
    C=C,
    tau=tau,
    sigma=sigma,
    x0=adjoint_pass(acq),
    maxiter=MAX_ITER
)


#######################################
# BM3D PnP ADMM
#######################################

eps = 5e-2

g = eps * 6.1e-2 * functional.BM3D(is_rgb=True)
C = linop.Identity(input_shape=input_size)

rho = 1.5 * eps * 10**-1

solver_BM3D = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[rho],
    x0=x_baseline,
    maxiter=MAX_ITER // 10,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={'tol': 1e-3, 'maxiter': 100})
)


#######################################
# DnCNN PnP ADMM
#######################################

eps = 1e0

g = eps * functional.DnCNN('6M')
C = linop.Identity(input_shape=input_size)

rho = 1e-2

solver_DnCNN = ADMM(
    f=f,
    g_list=[g],
    C_list=[C],
    rho_list=[rho],
    x0=x_baseline,
    maxiter=MAX_ITER // 5,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={'tol': 1e-3, 'maxiter': 100}),
    itstat_options={'display': True, 'period': 10}
)


#######################################
# Solving the problems
#######################################
# start = perf_counter()
# x_ADMM = solver_ADMM.solve()
# mid_1 = perf_counter()
# x_LADMM = solver_LADMM.solve()
# mid_2 = perf_counter()
# x_PDHG = solver_PDHG.solve()
# mid_3 = perf_counter()
# x_BM3D = solver_BM3D.solve()
mid_4 = perf_counter()
x_DnCNN = solver_DnCNN.solve()
end = perf_counter()


#######################################
# Solving the problems
#######################################

fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True)
ax[0][0].imshow(gt)
ax[0][0].set_title('Reference')
ax[0][1].imshow(acq, cmap='gray')
ax[0][1].set_title(f'Raw image')
ax[0][2].imshow(x_baseline)
ax[0][2].set_title(f'Baseline: {metric.psnr(gt, x_baseline):.2f} (dB)')
# ax[0][3].imshow(x_ADMM)
# ax[0][3].set_title(f'TV ADMM: {metric.psnr(gt, x_ADMM):.2f} (dB), {mid_1 - start:.1f}s')
# ax[1][0].imshow(x_LADMM)
# ax[1][0].set_title(f'TV LADMM: {metric.psnr(gt, x_LADMM):.2f} (dB), {mid_2 - mid_1:.1f}s')
# ax[1][1].imshow(x_PDHG)
# ax[1][1].set_title(f'PDHG: {metric.psnr(gt, x_PDHG):.2f} (dB), {mid_3 - mid_2:.1f}s')
# ax[1][2].imshow(x_BM3D)
# ax[1][2].set_title(f'BM3D PnP ADMM: {metric.psnr(gt, x_BM3D):.2f} (dB), {mid_4 - mid_3:.1f}s')
ax[1][3].imshow(x_DnCNN)
ax[1][3].set_title(f'DnCNN PnP ADMM: {metric.psnr(gt, x_DnCNN):.2f} (dB), {end - mid_4:.1f}s')
plt.show()