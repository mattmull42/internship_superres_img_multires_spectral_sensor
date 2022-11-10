from time import perf_counter
from colorama import Fore
from datetime import datetime

from src.pipeline_v1_class import *
from src.pipeline_v2_class import *


def run_pipeline_tests(input_paths, noise_level, pipeline_version, pipeline_parameters=None):
    print(Fore.YELLOW + f'###################### Beginning of the pipeline v{pipeline_version} tests ######################')

    create_output_dirs()

    with open(path.join('output', 'pipeline_log.csv'), 'a') as pipeline_log:
        csv_writer = csv.writer(pipeline_log)

        for cfa, binning in zip(['bayer', 'quad_bayer', 'quad_bayer', 'sparse_3'], [False, False, True, False]):
            if pipeline_version == 1:
                pipeline = Pipeline_v1(cfa, binning, noise_level)
            
            elif pipeline_version == 2:
                pipeline = Pipeline_v2(cfa, binning, noise_level, pipeline_parameters[0], pipeline_parameters[1])

            for input_path in input_paths:
                start = perf_counter()
                pipeline.run(input_path)
                duration = perf_counter() - start

                pipeline.save_output()

                if binning:
                    prefix = cfa.capitalize() + ' & binning'

                else:
                    prefix = cfa.capitalize()

                print(Fore.GREEN + prefix + f' pipeline v{pipeline_version} test passed in {duration:.2f} seconds on image {input_path}.')

                mse = pipeline.mse_error
                ssim = pipeline.ssim_error
                csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cfa, binning, noise_level, f'Pipeline v{pipeline_version}', input_path, f'{duration:.2f}', f'{mse:.4f}', f'{ssim:.4f}'])

        csv_writer.writerow([])

    print(Fore.YELLOW + f'######################### End of the pipeline v{pipeline_version} tests #########################')
