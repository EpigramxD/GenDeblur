import cv2 as cv
import numpy as np
from numpy.fft import fft2, ifft2


def get_dft(image, dft_type=cv.DFT_COMPLEX_OUTPUT):
    """
    Получить образ Фурье изображения
    :param image: изображение
    :param dft_type: вид DFT
    :return: DFT
    """
    #image_normalized = image / 255.0
    planes = [np.float32(image), np.zeros(image.shape, np.float32)]
    dft_ready = cv.merge(planes)
    cv.dft(dft_ready, dft_ready, flags=dft_type)
    return dft_ready


def invert_dft(dft):
    """
    Выполнить обратное преобразование Фурье
    :param dft: образ Фурье
    :return: результат инверсии
    """
    shape = (dft.shape[0], dft.shape[1], 1)
    planes = [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
    idft_result = cv.idft(dft)
    cv.split(idft_result, planes)
    return planes[0]


def get_dft_magnitude(dft):
    """
    Получить асболютную часть образа Фурье
    :param dft: образ Фурье
    :return: асболютная чать образа Фурье
    """
    correct_shape = (dft.shape[0], dft.shape[1], 1)
    planes = [np.zeros(correct_shape, np.float32), np.zeros(correct_shape, np.float32)]
    cv.split(dft, planes)
    # магнитуда находится в planes[0]
    cv.magnitude(planes[0], planes[1], planes[0])
    magnitude = planes[0]
    mat_of_ones = np.ones(magnitude.shape, dtype=magnitude.dtype)
    cv.add(mat_of_ones, magnitude, magnitude)
    cv.log(magnitude, magnitude)
    cv.normalize(magnitude, magnitude, 0, 1, cv.NORM_MINMAX)
    return magnitude


def freq_filter_1c(image, filter):
    """
    Частотная фильтрация для одного канала
    :param image: фильтруемое изображение
    :param psf: ядро свертки
    :return: результат фильтрации
    """
    image_copy = np.copy(image)
    image_fft = np.fft.fft2(image_copy)
    filter_fft = np.fft.fft2(filter, s=image.shape)

    fft_mul = image_fft * filter_fft
    fft_mul = np.abs(ifft2(fft_mul))

    result = np.zeros(fft_mul.shape)
    result = np.float32(result)

    fft_mul = np.float32(fft_mul)

    cv.normalize(fft_mul, result, 0.0, 1.0, cv.NORM_MINMAX)
    return result

