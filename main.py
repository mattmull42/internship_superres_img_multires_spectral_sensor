#!/usr/bin/env python3

import sys
from os import listdir
from time import time

from testing.adjoint_tests import *
from testing.pipeline_v1_tests import *
from testing.pipeline_v2_tests import *


def main(argv):
    if len(argv) == 1:
        input_names = listdir('input')

    else:
        input_names = argv[1:]

    run_adjoint_tests()

    run_pipeline_v1_tests(input_names)

    run_pipeline_v2_tests(input_names, 500)


if __name__ == "__main__":
    main(sys.argv)

    # input_name = 'tarkus.png'
    # x, spectral_stencil = initialize_input(input_name)
    # input_size = x.shape
    # cfa = 'quad_bayer'
    # binning = True
    # niter = 50

    # pipeline = Pipeline_v2(cfa, binning, niter)
    # TV = TV_operator(input_size)
    # TV_adj = TV_adjoint(input_size)

    # from cProfile import Profile
    # import pstats

    # with Profile()as pr:
    #     run_adjoint_tests()

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.dump_stats(filename='stats.prof')