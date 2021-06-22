import cv2 as cv
import numpy as np


def build_size_pyramid(image, kernel_size):
    """
    Строит пирамиду в виде словаря с ключем размера psf и соответствующего ресайзнутого изображения
    :param image: исходное размытое изображение
    :param kernel_size: максимальный размер ядра
    :return:
    """
    kernel_sizes = get_kernel_sizes(kernel_size)
    result = dict()
    for size in kernel_sizes:
        multiplier = size / kernel_size
        result[size] = cv.resize(image, None, fx=multiplier, fy=multiplier, interpolation=cv.INTER_AREA)
        result[size] = np.float32(result[size])
        cv.normalize(result[size], result[size], 0.0, 1.0, cv.NORM_MINMAX)
    return result


def get_kernel_sizes(kernel_size):
    """
    Получить размеры ядер для пирамиды в зависимости от максимального размера psf
    :param kernel_size: максимальный размер ядра
    :return: список размеров ядра
    """
    kernel_sizes = []

    step = 2
    current_kernel_size = 3
    while current_kernel_size <= kernel_size:
        kernel_sizes.append(current_kernel_size)
        current_kernel_size += step
        step *= 2

    if kernel_sizes[len(kernel_sizes) - 1] < kernel_size:
        kernel_sizes.append(kernel_size)

    return kernel_sizes