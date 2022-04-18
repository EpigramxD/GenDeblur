import multiprocessing as mp

import cv2

import utils.drawing as drawing
from utils.size_utils import *
from gen.genDeblurrer import GenDeblurrer
# константы
STAGNATION_POPULATION_COUNT = 20
UPSCALE_TYPE = "fill"
NO_REF_METRIC = "fourier"
REF_METRIC = "ssim"
DECONV_TYPE = "wiener"
SELECTION_ARGS = {"type" : "tournament", "k" : 3, "tournsize" : 5} # можно 5
CROSSOVER_ARGS = {"type" : "uniform", "probability" : 0.9}
MUTATION_ARGS = {"type" : "smart", "probability" : 0.1, "pos_probability" : 0.5} # можно 0.5
PYRAMID_ARGS = {"min_psf_size" : 3, "step" : 4, "max_psf_size" : 23}
ELITE_COUNT = 1
POPULATION_EXPAND_FACTOR = 40

# четкое изображение
sharp = cv.imread("../images/sharp/bstu1.jpg", cv.IMREAD_GRAYSCALE)
sharp = sharp ** (1/2.2)

# psf = drawing.draw_gaussian(sharp.shape, 3.0)
psf = cv.imread("../images/psfs/3.png", cv.IMREAD_GRAYSCALE)
# psf = ImgUtils.pad_to_shape(psf, sharp.shape)

blurred = ImgUtils.freq_filter(sharp, psf)
# blurred = cv2.GaussianBlur(sharp, (11, 11), 0)
cv.normalize(blurred, blurred, 0.0, 1.0, cv.NORM_MINMAX)

if __name__ == '__main__':
    mp.freeze_support()

    mp_manager = mp.Manager()
    deblurrer = GenDeblurrer(STAGNATION_POPULATION_COUNT,
                             ref_metric_type=REF_METRIC,
                             no_ref_metric_type=NO_REF_METRIC,
                             deconv_type=DECONV_TYPE,
                             selection_args=SELECTION_ARGS,
                             crossover_args=CROSSOVER_ARGS,
                             mutation_args=MUTATION_ARGS,
                             pyramid_args=PYRAMID_ARGS,
                             elite_count=ELITE_COUNT,
                             population_expand_factor=POPULATION_EXPAND_FACTOR,
                             upscale_type=UPSCALE_TYPE,
                             multiprocessing_manager=mp_manager)
    deblurrer.deblur(blurred)

