import cv2 as cv
from numpy.fft import fft2, ifft2
import numpy as np
from skimage import color, data, restoration
from utils.drawing import draw_line
from utils.metric import *


def do_RL_deconv_1c(image, psf, iterations, clip=True):
    if len(image.shape) == 2:
        image_ = np.float32(image)
        cv.normalize(image_, image_, 0.0, 1.0, cv.NORM_MINMAX)
        result = restoration.richardson_lucy(image_, psf.copy(), iterations=iterations, clip=clip)
        result = np.float32(result)
        cv.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        return result
    else:
        raise AttributeError("Wrong image format, image should be grayscale")


def do_RL_deconv_3c(image, psf, iterations, clip=True):
    if len(image.shape) == 3:
        image_ = np.float32(image)
        channels = list(cv.split(image_))
        channels[:] = [do_RL_deconv_1c(channel, psf.copy(), iterations, clip) for channel in channels]
        result = cv.merge(channels)
        result = np.float32(result)
        cv.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        return result
    else:
        raise AttributeError("Wrong image format, image should be RGB")

def do_RL_deconv(image, psf, iterations, clip=True):
    if len(image.shape) == 3:
        return do_RL_deconv_3c(image, psf, iterations, clip)
    else:
        return do_RL_deconv_1c(image, psf, iterations, clip)

def do_weiner_deconv_1c(image, psf, K):
    dummy = np.copy(image)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(psf, s=image.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    result = np.zeros(dummy.shape)
    result = np.float32(result)
    dummy = np.float32(dummy)
    cv.normalize(dummy, result, 0.0, 1.0, cv.NORM_MINMAX)
    return result

# TODO do_weiner_deconv_3c
