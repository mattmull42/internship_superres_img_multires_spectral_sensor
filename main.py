from src.pipeline_class import *


input_name = 'tarkus.png'

pipeline_bayer = Pipeline('bayer', False)
pipeline_quad = Pipeline('quad_bayer', False)
pipeline_binning = Pipeline('quad_bayer', True)

pipeline_bayer.run(input_name)
pipeline_quad.run(input_name)
pipeline_binning.run(input_name)