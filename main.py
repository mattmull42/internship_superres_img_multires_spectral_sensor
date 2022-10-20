#!/usr/bin/env python3

from os import listdir

from testing.adjoint_tests import *
from testing.batch_tests import *
from testing.circulant_test import *
from testing.pipeline_tests import *


NOISE_LEVEL = 0
NITER = 1000
SIGMA = 50
EPS = 0.001
BOX_FLAG = True
INPUT_DIR = 'input'
BATCH_DIR = 'input_batch'
BATCH_ARRAY = [path.join(BATCH_DIR, image_name) for image_name in listdir(BATCH_DIR)]


def main():
    run_adjoint_tests()

    # run_circulant_tests()

    # run_pipeline_tests(input_paths, NOISE_LEVEL, pipeline_version=1)
    # run_pipeline_tests(input_paths, NOISE_LEVEL, pipeline_version=2, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])

    # run_batch_tests(BATCH_ARRAY, NOISE_LEVEL, pipeline_version=1, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])
    # run_batch_tests(BATCH_ARRAY, NOISE_LEVEL, pipeline_version=2, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])

    CFA = 'sparse_3'
    BINNING = CFA == 'quad_bayer'

    from PIL import ImageOps

    x, spectral_stencil = initialize_input('input/01690.png')
    # x = initialize_inverse_input('input/ProxOnyx_0.tiff')
    # x = np.array(ImageOps.equalize(Image.fromarray(((x / np.max(x)) * 255).astype(np.uint8)))) / 255

    input_size = np.append(x.shape, 3)
    input_size = x.shape

    forward_op = Forward_operator(CFA, BINNING, 0, input_size, spectral_stencil)
    baseline_op = Inverse_problem(CFA, BINNING, forward_op.get_parameters())
    # admm_op = Inverse_problem_ADMM(CFA, BINNING, 0, input_size, spectral_stencil, NITER, SIGMA, EPS, BOX_FLAG)

    res = forward_op(x).asarray()
    res_baseline = baseline_op(res)
    # res_admm = admm_op(x)

    baseline_op.save_output('ProxOnyx_0')
    # admm_op.save_output('ProxOnyx_0')

    plt.imshow(res_baseline)
    plt.show()

    # gt = np.asarray(Image.open('reu/GT.png')) / 255
    # baseline = np.asarray(Image.open('reu/baseline.png'))[:, :, :-1] / 255
    # admm = np.asarray(Image.open('reu/ADMM.png'))[:, :, :-1] / 255
    # pipnet_bayer = np.asarray(Image.open('reu/PIPNet bayer weights.png')) / 255
    # pipnet_quad = np.asarray(Image.open('reu/PIPNet quad weights.png')) / 255

    # print('SSIM :')
    # print(f'    Baseline : {structural_similarity(gt, baseline, channel_axis=2):.4f}')
    # print(f'    ADMM : {structural_similarity(gt, admm, channel_axis=2):.4f}')
    # print(f'    PIPNet with bayer weights : {structural_similarity(gt, pipnet_bayer, channel_axis=2):.4f}')
    # print(f'    PIPNet with quad bayer weights : {structural_similarity(gt, pipnet_quad, channel_axis=2):.4f}')
    # print('MSE :')
    # print(f'    Baseline : {mean_squared_error(gt, baseline):.4f}')
    # print(f'    ADMM : {mean_squared_error(gt, admm):.4f}')
    # print(f'    PIPNet with bayer weights : {mean_squared_error(gt, pipnet_bayer):.4f}')
    # print(f'    PIPNet with quad bayer weights : {mean_squared_error(gt, pipnet_quad):.4f}')


if __name__ == "__main__":
    main()
