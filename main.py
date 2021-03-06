#!/usr/bin/env python3

import sys
from os import listdir

from testing.adjoint_tests import *
from testing.circulant_test import *
from testing.batch_tests import *
from testing.pipeline_tests import *


NOISE_LEVEL = 0
NITER = 1000
SIGMA = 50
EPS = 0.001
BOX_FLAG = True
INPUT_DIR = 'input'
BATCH_DIR = 'input_batch'
BATCH_ARRAY = [path.join(BATCH_DIR, image_name) for image_name in listdir(BATCH_DIR)]


def main(argv):
    if len(argv) == 1:
        input_paths = [path.join(INPUT_DIR, image_name) for image_name in listdir(INPUT_DIR)]

    else:
        input_paths = argv[1:]

    # run_adjoint_tests()

    # run_circulant_tests()

    # run_pipeline_tests(input_paths, NOISE_LEVEL, pipeline_version=1)
    # run_pipeline_tests(input_paths, NOISE_LEVEL, pipeline_version=2, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])

    # run_batch_tests(BATCH_ARRAY, NOISE_LEVEL, pipeline_version=1, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])
    # run_batch_tests(BATCH_ARRAY, NOISE_LEVEL, pipeline_version=2, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])

    x, spectral_stencil = initialize_input('input/01690.png')

    forward_op = Forward_operator('bayer', x.shape, spectral_stencil, False, 0)
    # forward_op(x)
    # forward_op.save_output('input/01690.png')

    # plt.imshow(forward_op.apply_matrix_operator(x), cmap="gray")
    # plt.show()

    matrix = forward_op.get_matrix_operator()
    matrix = matrix @ matrix.T


if __name__ == "__main__":
    main(sys.argv)
