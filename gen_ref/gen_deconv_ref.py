from utils.size_utils import *
from gen.genDeblurrer import GenDeblurrer
# константы
STAGNATION_POPULATION_COUNT = 15
UPSCALE_TYPE = "pad"
NO_REF_METRIC = "fourier"
REF_METRIC = "ssim"
DECONV_TYPE = "wiener"
SELECTION_ARGS = {"type" : "tournament", "k" : 3, "tournsize" : 2}
CROSSOVER_ARGS = {"type" : "uniform", "probability" : 0.9}
MUTATION_ARGS = {"type" : "smart", "probability" : 0.1, "pos_probability" : 0.5}
ELITE_COUNT = 1
MIN_PSF_SIZE = 3
STEP = 2
MAX_PSF_SIZE = 23

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
                         min_psf_size=MIN_PSF_SIZE,
                         step=STEP,
                         max_psf_size=MAX_PSF_SIZE,
                         elite_count=ELITE_COUNT)

deblurrer.deblur(sharp, blurred)