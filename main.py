#!/usr/bin/env python3

import sys
from os import listdir

from src.pipeline_class import *
from src.utilities.plots import *


def main(argv):
    pipeline_bayer = Pipeline('bayer', False)
    pipeline_quad = Pipeline('quad_bayer', False)
    pipeline_binning = Pipeline('quad_bayer', True)

    if len(argv) == 1:
        input_names = listdir('input')

    else:
        input_names = argv[1:]

    for input_name in input_names:
        pipeline_bayer.run(input_name)
        pipeline_quad.run(input_name)
        pipeline_binning.run(input_name)


if __name__ == "__main__":
    main(sys.argv)