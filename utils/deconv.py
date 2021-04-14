import cv2 as cv
import numpy as np
from skimage import color, data, restoration
from utils.drawing import draw_line
from utils.metric import *


def do_RL_deconv_1c(image, psf, iterations, clip=True):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_ = np.float32(image)
        cv.normalize(image_, image_, 0.0, 1.0, cv.NORM_MINMAX)
        result = restoration.richardson_lucy(image_, psf, iterations=iterations, clip=clip)
        cv.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        return result
    else:
        raise AttributeError("Wrong image format, image should be grayscale")


def do_RL_deconv_3c(image, psf, iterations, clip=True):
    if image.shape[2] == 3:
        image_ = np.float32(image)
        channels = list(cv.split(image_))
        channels[:] = [do_RL_deconv_1c(channel, psf, iterations, clip) for channel in channels]
        result = cv.merge(channels)
        cv.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        return result
    else:
        raise AttributeError("Wrong image format, image should be RGB")