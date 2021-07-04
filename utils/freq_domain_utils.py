import cv2 as cv
import numpy as np
import copy
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


def freq_filter_1c(image, filter):
    """
    Частотная фильтрация для одного канала
    :param image: фильтруемое изображение
    :param psf: ядро свертки
    :return: результат фильтрации
    """
    kernel = pad_to_shape(filter, image.shape)
    image_copy = np.copy(image)
    image_fft = np.fft.fft2(image_copy)
    filter_fft = np.fft.fft2(kernel)

    fft_mul = image_fft * filter_fft
    fft_mul = np.abs(ifft2(fft_mul))

    result = np.zeros(fft_mul.shape)
    result = np.float32(result)

    fft_mul = np.float32(fft_mul)

    cv.normalize(fft_mul, result, 0.0, 1.0, cv.NORM_MINMAX)

    return np.fft.fftshift(result)

