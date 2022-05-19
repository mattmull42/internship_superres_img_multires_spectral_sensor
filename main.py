#!/usr/bin/env python3

import sys
from os import listdir

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

    run_pipeline_v2_tests(input_names, 300)


if __name__ == "__main__":
    main(sys.argv)