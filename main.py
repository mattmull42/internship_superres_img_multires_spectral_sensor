#!/usr/bin/env python3

import sys
from os import listdir

from testing.adjoint_tests import *
from testing.circulant_test import *
from testing.batch_tests import *
from testing.pipeline_tests import *


NOISE_LEVEL = 0
NITER = 2000
SIGMA = 50
EPS = 0.001
BOX_FLAG = True
INPUT_DIR = 'input'
BATCH_DIR = 'input_batch'
BATCH_ARRAY = [path.join(BATCH_DIR, image_name) for image_name in listdir(BATCH_DIR)]


def main(argv):
    # run_adjoint_tests()

    # run_circulant_tests()

    # run_pipeline_tests(input_paths, NOISE_LEVEL, pipeline_version=1)
    # run_pipeline_tests(input_paths, NOISE_LEVEL, pipeline_version=2, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])

    # run_batch_tests(BATCH_ARRAY, NOISE_LEVEL, pipeline_version=1, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])
    # run_batch_tests(BATCH_ARRAY, NOISE_LEVEL, pipeline_version=2, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])

    CFA = 'sparse_3'
    BINNING = CFA == 'quad_bayer'

    x, spectral_stencil = initialize_input('input/01690.png')

    forward_op = Forward_operator(CFA, BINNING, 0, x.shape, spectral_stencil)
    baseline_op = Inverse_problem(CFA, BINNING, forward_op.get_parameters())
    # ADMM_op = Inverse_problem_ADMM(CFA, BINNING, 0, x.shape, spectral_stencil, NITER, SIGMA, EPS, BOX_FLAG)

    res = forward_op(x)

    res = baseline_op(res)

    plt.imshow(res)
    plt.show()

    # gt = np.asarray(Image.open('reu/GT.png')) / 255
    # admm = np.asarray(Image.open('reu/ADMM.png'))[:, :, :-1] / 255
    # pipnet_bayer = np.asarray(Image.open('reu/PIPNet bayer weights.png')) / 255
    # pipnet_quad = np.asarray(Image.open('reu/PIPNet quad weights.png')) / 255

    # print('SSIM :')
    # print(f'    ADMM : {structural_similarity(gt, admm, channel_axis=2):.4f}')
    # print(f'    PIPNet with bayer weights : {structural_similarity(gt, pipnet_bayer, channel_axis=2):.4f}')
    # print(f'    PIPNet with quad bayer weights : {structural_similarity(gt, pipnet_quad, channel_axis=2):.4f}')
    # print('MSE :')
    # print(f'    ADMM : {mean_squared_error(gt, admm):.4f}')
    # print(f'    PIPNet with bayer weights : {mean_squared_error(gt, pipnet_bayer):.4f}')
    # print(f'    PIPNet with quad bayer weights : {mean_squared_error(gt, pipnet_quad):.4f}')


if __name__ == "__main__":
    main(sys.argv)
