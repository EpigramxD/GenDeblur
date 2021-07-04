import cv2 as cv
import numpy as np
from utils.freq_domain_utils import freq_filter_1c
import copy


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
            blurred_resized = freq_filter_1c(sharp_resized, psf_resized)
        else:
            sharp_resized = copy.deepcopy(sharp)
            psf_resized = copy.deepcopy(psf)
            blurred_resized = freq_filter_1c(sharp_resized, psf_resized)

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


def pad_to_shape(image, shape):
    """
    Расширить изображение до нужной формы нулями (черными рамками по краям)
    :param image: расширяемое изображение
    :param shape: новая форма - tuple (высота, ширина)
    :return: рсширенное изображение
    """
    if shape[0] == image.shape[0] and shape[1] == image.shape[1]:
        return copy.deepcopy(image)

    elif shape[0] < image.shape[0] or shape[1] < image.shape[1]:
        raise AttributeError("Новый размер должен быть меньше размера изображения")

    center_y = int(shape[0] / 2)
    center_x = int(shape[1] / 2)

    half_height = int(image.shape[0] / 2)
    half_width = int(image.shape[1] / 2)

    x_index_from = center_x - half_width
    x_index_to = center_x + image.shape[1] - half_width

    y_index_from = center_y - half_height
    y_index_to = center_y + image.shape[0] - half_height

    result = np.zeros(shape, np.float32)
    result[y_index_from:y_index_to, x_index_from:x_index_to] = copy.deepcopy(image)
    cv.normalize(result, result, 0.0, 1.0, cv.NORM_MINMAX)
    return result