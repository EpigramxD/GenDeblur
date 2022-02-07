from utils.size_utils import *
from gen.genDeblurrer import GenDeblurrer
# константы
STAGNATION_POPULATION_COUNT = 50
UPSCALE_TYPE = "pad"
NO_REF_METRIC = "fourier"
REF_METRIC = "ssim"
DECONV_TYPE = "wiener"
SELECTION_ARGS = {"type" : "tournament", "k" : 3, "tournsize" : 2}
CROSSOVER_ARGS = {"type" : "uniform", "probability" : 0.9}
MUTATION_ARGS = {"type" : "smart", "probability" : 0.1, "pos_probability" : 0.5}
PYRAMID_ARGS = {"min_psf_size" : 3, "step" : 2, "max_psf_size" : 23, "inter_type": cv.INTER_AREA}
ELITE_COUNT = 1
POPULATION_EXPAND_FACTOR = 10

# четкое изображение
sharp = cv.imread("../images/sharp/bstu2.jpg", cv.IMREAD_GRAYSCALE)
#sharp = sharp ** (1/2.2)

psf = cv.imread("../images/psfs/2.png", cv.IMREAD_GRAYSCALE)
psf = ImgUtils.pad_to_shape(psf, sharp.shape)
blurred = ImgUtils.freq_filter(sharp, psf)
cv.normalize(blurred, blurred, 0.0, 1.0, cv.NORM_MINMAX)


deblurrer = GenDeblurrer(STAGNATION_POPULATION_COUNT,
                         ref_metric_type=REF_METRIC,
                         no_ref_metric_type=NO_REF_METRIC,
                         deconv_type=DECONV_TYPE,
                         selection_args=SELECTION_ARGS,
                         crossover_args=CROSSOVER_ARGS,
                         mutation_args=MUTATION_ARGS,
                         pyramid_args=PYRAMID_ARGS,
                         elite_count=ELITE_COUNT,
                         population_expand_factor=POPULATION_EXPAND_FACTOR)

deblurrer.deblur(sharp, blurred)