#!/usr/bin/env python3

import sys
from os import listdir
import csv
from tqdm import tqdm

from src.pipeline_class import *


def main(argv):
    pipeline_bayer = Pipeline('bayer', False)
    pipeline_quad = Pipeline('quad_bayer', False)
    pipeline_binning = Pipeline('quad_bayer', True)

    if len(argv) == 1:
        input_names = listdir('input')

    else:
        input_names = argv[1:]

    create_output_dirs()
    
    with open(path.join('output', 'errors_log.csv'), 'w') as errors_log:
        csvwriter = csv.writer(errors_log)
        csvwriter.writerow(['Name', 'MSE red error', 'MSE green error', 'MSE blue error', 'SSIM red error', 'SSIM green error', 'SSIM blue error'])

        for input_name in tqdm(input_names, desc='Processed input :'):
            pipeline_bayer.run(input_name)
            pipeline_quad.run(input_name)
            pipeline_binning.run(input_name)

            mse_bayer = pipeline_bayer.mse_errors
            mse_quad = pipeline_quad.mse_errors
            mse_binning = pipeline_binning.mse_errors
            ssim_bayer = pipeline_bayer.ssim_errors
            ssim_quad = pipeline_quad.ssim_errors
            ssim_binning = pipeline_binning.ssim_errors
            csvwriter.writerow([input_name[:-4] + '_bayer'] + mse_bayer + ssim_bayer)
            csvwriter.writerow([input_name[:-4] + '_quad_bayer'] + mse_quad + ssim_quad)
            csvwriter.writerow([input_name[:-4] + '_quad_bayer_binning'] + mse_binning + ssim_binning)


if __name__ == "__main__":
    main(sys.argv)