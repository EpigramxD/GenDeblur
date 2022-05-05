import multiprocessing as mp

import cv2 as cv

from gen.genDeblurrer import GenDeblurrer
from utils.imgUtils import ImgUtils

# четкое изображение
sharp = cv.imread("images/sharp/lena.png", cv.IMREAD_GRAYSCALE)
sharp = sharp ** (1/2.2)

# PSF
# psf = drawing.draw_gaussian(sharp.shape, 3.0)
psf = cv.imread("images/psfs/3.png", cv.IMREAD_GRAYSCALE)
# размытие
blurred = ImgUtils.freq_filter(sharp, psf)
# наложение шума
# blurred = misc.get_noisy_image(blurred, 0.00172135)
cv.normalize(blurred, blurred, 0.0, 1.0, cv.NORM_MINMAX)

if __name__ == '__main__':
    mp.freeze_support()

    mp_manager = mp.Manager()
    deblurrer = GenDeblurrer("configuration.yaml", mp_manager)
    deblurred_img, psf = deblurrer.deblur(blurred)

