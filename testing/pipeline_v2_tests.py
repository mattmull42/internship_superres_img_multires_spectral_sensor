from time import time
from colorama import Fore
from tqdm import tqdm
from datetime import datetime

from src.pipeline_v2_class import *


def run_pipeline_v2_tests(input_names, niter):
    print(Fore.YELLOW + '###################### Beginning of the pipeline V2 tests ######################')

    create_output_dirs()

    pipeline_bayer = Pipeline_v2('bayer', False, niter)
    pipeline_quad = Pipeline_v2('quad_bayer', False, niter)
    pipeline_binning = Pipeline_v2('quad_bayer', True, niter)

    with open(path.join('output', 'errors_log.csv'), 'a') as errors_log:
        csvwriter = csv.writer(errors_log)

        for input_name in tqdm(input_names, desc='Processed inputs ', colour='green'):
            start = time()
            pipeline_bayer.run(input_name)
            duration = time() - start
            tqdm.write(Fore.GREEN + f'Bayer pipeline test passed in {duration:.2f} seconds on image {input_name}.')

            mse_bayer = pipeline_bayer.mse_errors
            ssim_bayer = pipeline_bayer.ssim_errors
            csvwriter.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Pipeline V2', input_name[:-4] + '_bayer', np.mean(mse_bayer), np.mean(ssim_bayer)] + mse_bayer + ssim_bayer)
            
            start = time()
            pipeline_quad.run(input_name)
            duration = time() - start
            tqdm.write(Fore.GREEN + f'Quad-Bayer pipeline test passed in {duration:.2f} seconds on image {input_name}.')

            mse_quad = pipeline_quad.mse_errors
            ssim_quad = pipeline_quad.ssim_errors
            csvwriter.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Pipeline V2', input_name[:-4] + '_quad_bayer', np.mean(mse_quad), np.mean(ssim_quad)] + mse_quad + ssim_quad)

            start = time()
            pipeline_binning.run(input_name)
            duration = time() - start
            tqdm.write(Fore.GREEN + f'Quad-Bayer & binning pipeline test passed in {duration:.2f} seconds on image {input_name}.')

            mse_binning = pipeline_binning.mse_errors
            ssim_binning = pipeline_binning.ssim_errors
            csvwriter.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Pipeline V2', input_name[:-4] + '_quad_bayer_binning', np.mean(mse_binning), np.mean(ssim_binning)] + mse_binning + ssim_binning)

            csvwriter.writerow([])

    print(Fore.YELLOW + '######################### End of the pipeline V2 tests #########################')