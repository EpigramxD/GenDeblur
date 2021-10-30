import copy
import numpy as np


def crop_image(image, crop_width_size, crop_height_size):
    """
    Обрезает изображение по краям
    :param image: изображение
    :param crop_width_size: размер отсечения по бокам
    :param crop_height_size: размер отсечения сверху и снизу
    :return: обрезанное изображение
    """
    width = image.shape[1]
    height = image.shape[0]

    h_start = crop_height_size
    h_finish = height - crop_height_size

    w_start = crop_width_size
    w_finish = width - crop_width_size

    return copy.deepcopy(image[h_start:h_finish, w_start:w_finish])


def get_noisy_image(image):
    row, col = image.shape
    mean = 0
    var = 0.002
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy
