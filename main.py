#!/usr/bin/env python3

import sys
from os import listdir

from testing.adjoint_tests import *
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
    if len(argv) == 1:
        input_paths = [path.join(INPUT_DIR, image_name) for image_name in listdir(INPUT_DIR)]

    else:
        input_paths = argv[1:]

    # run_adjoint_tests()

    # run_pipeline_tests(input_paths, NOISE_LEVEL, pipeline_version=1)
    run_pipeline_tests(input_paths, NOISE_LEVEL, pipeline_version=2, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])

    # for noise_level in [0, 5, 10, 15]:
    #     run_batch_tests(BATCH_ARRAY, noise_level, pipeline_version=1, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])
        # run_batch_tests(BATCH_ARRAY, noise_level, pipeline_version=2, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])

    # run_batch_tests(BATCH_ARRAY, NOISE_LEVEL, pipeline_version=1, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])
    # run_batch_tests(BATCH_ARRAY, NOISE_LEVEL, pipeline_version=2, pipeline_parameters=[NITER, SIGMA, EPS, BOX_FLAG])


if __name__ == "__main__":
    main(sys.argv)
