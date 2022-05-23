from time import perf_counter
from colorama import Fore
from datetime import datetime

from src.pipeline_v1_class import *


def run_pipeline_v1_tests(input_names):
    print(Fore.YELLOW + '###################### Beginning of the pipeline V1 tests ######################')

    create_output_dirs()

    pipeline_bayer = Pipeline_v1('bayer', False)
    pipeline_quad = Pipeline_v1('quad_bayer', False)
    pipeline_binning = Pipeline_v1('quad_bayer', True)

    with open(path.join('output', 'errors_log.csv'), 'a') as errors_log:
        csvwriter = csv.writer(errors_log)

        for input_name in input_names:
            start = perf_counter()
            pipeline_bayer.run(input_name)
            duration = perf_counter() - start
            print(Fore.GREEN + f'Bayer pipeline test passed in {duration:.2f} seconds on image {input_name}.')

            mse_bayer = pipeline_bayer.mse_errors
            ssim_bayer = pipeline_bayer.ssim_errors
            csvwriter.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Pipeline V1', input_name[:-4] + '_bayer', f'{duration:.2f}', np.mean(mse_bayer), np.mean(ssim_bayer)] + mse_bayer + ssim_bayer)
            
            start = perf_counter()
            pipeline_quad.run(input_name)
            duration = perf_counter() - start
            print(Fore.GREEN + f'Quad-Bayer pipeline test passed in {duration:.2f} seconds on image {input_name}.')

            mse_quad = pipeline_quad.mse_errors
            ssim_quad = pipeline_quad.ssim_errors
            csvwriter.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Pipeline V1', input_name[:-4] + '_quad_bayer', f'{duration:.2f}', np.mean(mse_quad), np.mean(ssim_quad)] + mse_quad + ssim_quad)

            start = perf_counter()
            pipeline_binning.run(input_name)
            duration = perf_counter() - start
            print(Fore.GREEN + f'Quad-Bayer & binning pipeline test passed in {duration:.2f} seconds on image {input_name}.')

            mse_binning = pipeline_binning.mse_errors
            ssim_binning = pipeline_binning.ssim_errors
            csvwriter.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Pipeline V1', input_name[:-4] + '_quad_bayer_binning', f'{duration:.2f}', np.mean(mse_binning), np.mean(ssim_binning)] + mse_binning + ssim_binning)

            csvwriter.writerow([])

    print(Fore.YELLOW + '######################### End of the pipeline V1 tests #########################')