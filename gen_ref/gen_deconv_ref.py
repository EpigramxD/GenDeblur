from utils.size_utils import *
from gen.genDeblurrer import GenDeblurrer
# константы
STAGNATION_POPULATION_COUNT = 15
UPSCALE_TYPE = "pad"
NO_REF_METRIC = "fourier"
REF_METRIC = "ssim"
DECONV_TYPE = "wiener"
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.1
POS_PROBABILITY = 0.5
ELITE_COUNT = 1
MIN_KERNEL_SIZE = 3
STEP = 2
MAX_KERNEL_SIZE = 23

# четкое изображение
sharp = cv.imread("../images/sharp/bstu2.jpg", cv.IMREAD_GRAYSCALE)
#sharp = sharp ** (1/2.2)
sharp = np.float32(sharp)
cv.normalize(sharp, sharp, 0.0, 1.0, cv.NORM_MINMAX)

psf = cv.imread("../images/psfs/2.png", cv.IMREAD_GRAYSCALE)
psf = ImgUtils.pad_to_shape(psf, sharp.shape)
blurred = ImgUtils.freq_filter(sharp, psf)
cv.normalize(blurred, blurred, 0.0, 1.0, cv.NORM_MINMAX)


deblurrer = GenDeblurrer(STAGNATION_POPULATION_COUNT,
                      REF_METRIC,
                      NO_REF_METRIC,
                      DECONV_TYPE,
                      CROSSOVER_PROBABILITY,
                      MUTATION_PROBABILITY,
                      POS_PROBABILITY,
                      MIN_KERNEL_SIZE,
                      STEP,
                      MAX_KERNEL_SIZE,
                      ELITE_COUNT)

deblurrer.deblur(sharp, blurred)