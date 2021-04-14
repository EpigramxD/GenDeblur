import cv2 as cv
import numpy as np


def get_dft(image, dft_type=cv.DFT_COMPLEX_OUTPUT):
    #image_normalized = image / 255.0
    planes = [np.float32(image), np.zeros(image.shape, np.float32)]
    dft_ready = cv.merge(planes)
    cv.dft(dft_ready, dft_ready, flags=dft_type)
    return dft_ready


def invert_dft(dft, shape=None):
    shape = (dft.shape[0], dft.shape[1], 1)
    planes = [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
    idft_result = cv.idft(dft)
    cv.split(idft_result, planes)
    return planes[0]


def get_dft_magnitude(dft):
    correct_shape = (dft.shape[0], dft.shape[1], 1)
    planes = [np.zeros(correct_shape, np.float32), np.zeros(correct_shape, np.float32)]
    cv.split(dft, planes)
    # magnitude will be in planes[0]
    cv.magnitude(planes[0], planes[1], planes[0])
    magnitude = planes[0]
    mat_of_ones = np.ones(magnitude.shape, dtype=magnitude.dtype)
    cv.add(mat_of_ones, magnitude, magnitude)
    cv.log(magnitude, magnitude)
    cv.normalize(magnitude, magnitude, 0, 1, cv.NORM_MINMAX)
    return magnitude