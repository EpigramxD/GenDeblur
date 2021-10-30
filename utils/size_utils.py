import copy

import cv2 as cv
import numpy as np

from utils.filteringUtils import FilteringUtils


def build_size_pyramid(image, max_kernel_size):
    """
    Строит пирамиду в виде словаря с ключем размера psf и соответствующего ресайзнутого изображения
    :param image: исходное размытое изображение
    :param max_kernel_size: максимальный размер ядра
    :return:
    """
    kernel_sizes = get_kernel_sizes(max_kernel_size)
    result = dict()
    for size in kernel_sizes:
        multiplier = size / max_kernel_size
        result[size] = cv.resize(image, None, fx=multiplier, fy=multiplier, interpolation=cv.INTER_AREA)
        result[size] = np.float32(result[size])
        cv.normalize(result[size], result[size], 0.0, 1.0, cv.NORM_MINMAX)
    return result


def build_double_size_pyramid(sharp, blurred, max_kernel_size, inter_type=cv.INTER_AREA):
    """
    Строит пирамиду, но только уже для пар четкого и размытого изображения
    :param sharp: четкое изображение
    :param blurred: рамызтое изображение
    :param max_kernel_size: максимальный размер ядра
    :param inter_type: вид рессайза
    :return: пирамида
    """
    kernel_sizes = get_kernel_sizes(max_kernel_size)
    result = dict()
    for size in kernel_sizes:
        multiplier = size / max_kernel_size
        if multiplier != 1:
            sharp_resized = cv.resize(sharp, None, fx=multiplier, fy=multiplier, interpolation=inter_type)
            blurred_resized = cv.resize(blurred, None, fx=multiplier, fy=multiplier, interpolation=inter_type)
        else:
            sharp_resized = copy.deepcopy(sharp)
            blurred_resized = copy.deepcopy(blurred)

        sharp_resized = np.float32(sharp_resized)
        cv.normalize(sharp_resized, sharp_resized, 0.0, 1.0, cv.NORM_MINMAX)

        blurred_resized = np.float32(blurred_resized)
        cv.normalize(blurred_resized, blurred_resized, 0.0, 1.0, cv.NORM_MINMAX)

        result[size] = (sharp_resized, blurred_resized)
    return result


def build_double_size_pyramid_new(sharp, psf, max_kernel_size, inter_type=cv.INTER_AREA):
    """
    Строит пирамиду, но только уже для пар четкого и размытого изображения
    :param sharp: четкое изображение
    :param blurred: рамызтое изображение
    :param max_kernel_size: максимальный размер ядра
    :param inter_type: вид рессайза
    :return: пирамида
    """
    kernel_sizes = get_kernel_sizes(max_kernel_size)
    result = dict()
    for size in kernel_sizes:
        multiplier = size / max_kernel_size
        if multiplier != 1:
            sharp_resized = cv.resize(sharp, None, fx=multiplier, fy=multiplier, interpolation=inter_type)
            psf_resized = cv.resize(psf, None, fx=multiplier, fy=multiplier, interpolation=inter_type)
            blurred_resized = FilteringUtils.freq_filter(sharp_resized, psf_resized)
        else:
            sharp_resized = copy.deepcopy(sharp)
            psf_resized = copy.deepcopy(psf)
            blurred_resized = FilteringUtils.freq_filter(sharp_resized, psf_resized)

        sharp_resized = np.float32(sharp_resized)
        cv.normalize(sharp_resized, sharp_resized, 0.0, 1.0, cv.NORM_MINMAX)

        blurred_resized = np.float32(blurred_resized)
        cv.normalize(blurred_resized, blurred_resized, 0.0, 1.0, cv.NORM_MINMAX)

        result[size] = (sharp_resized, blurred_resized)
    return result


def get_kernel_sizes(max_kernel_size):
    """
    Получить размеры ядер для пирамиды в зависимости от максимального размера psf, начальный размер ядра - 3x3
    :param max_kernel_size: максимальный размер ядра
    :return: список размеров ядра
    """
    kernel_sizes = []

    step = 2
    current_kernel_size = 3
    while current_kernel_size <= max_kernel_size:
        kernel_sizes.append(current_kernel_size)
        current_kernel_size += step
        #step *= 2

    if kernel_sizes[len(kernel_sizes) - 1] < max_kernel_size:
        kernel_sizes.append(max_kernel_size)
    return kernel_sizes