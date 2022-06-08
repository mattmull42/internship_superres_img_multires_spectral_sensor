from colorama import Fore
from datetime import datetime
from tqdm import tqdm

from src.pipeline_v1_class import *
from src.pipeline_v2_class import *


def run_batch_tests(input_paths, noise_level, pipeline_version, pipeline_parameters=None):
    print(Fore.YELLOW + '######################### Beginning of the batch tests #########################')
    create_output_dirs()

    with open(path.join('output', 'batch_log.csv'), 'a') as batch_log:
        for cfa, binning in zip(['bayer', 'quad_bayer', 'quad_bayer'], [False, False, True]):
            if pipeline_version == 1:
                pipeline = Pipeline_v1(cfa, binning)

            elif pipeline_version == 2:
                pipeline = Pipeline_v2(cfa, binning, pipeline_parameters[0], pipeline_parameters[1], pipeline_parameters[2], pipeline_parameters[3])

            mse, ssim = [], []

            if binning:
                prefix = cfa.capitalize() + ' & binning '

            else:
                prefix = cfa.capitalize() + ' '

            for input_path in tqdm(input_paths, desc=prefix, colour='blue'):
                pipeline.run(input_path, noise_level)

                mse.append(np.mean(pipeline.mse_errors))
                ssim.append(np.mean(pipeline.ssim_errors))

            csv_writer = csv.writer(batch_log)
            csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cfa, binning, noise_level, f'Pipeline v{pipeline_version}', f'{np.mean(mse):.4f}', f'{np.mean(ssim):.4f}'])

        csv_writer.writerow([])

    print(Fore.YELLOW + '############################ End of the batch tests ############################')