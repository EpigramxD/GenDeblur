import copy
import numpy as np
import cv2 as cv

# TODO: вынести
def get_dark_channel(image, size):
    """
    Получить "темный" канал изображения
    :param image: изображение
    :param size: размер ядра
    :return: "темный" канал
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    dark_channel = image

    if len(image.shape) == 3:
        b, g, r = cv.split(image)
        dark_channel = cv.min(cv.min(r, g), b)

    dark_channel = cv.erode(dark_channel, kernel)
    return dark_channel


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


def grad_tv_1c(img, epsilon=1e-3):
    # x_forward
    x_f = copy.copy(img)
    x_f[0:img.shape[0] - 1, :] = copy.copy(img[1:, :])

    # y_forward
    y_f = copy.copy(img)
    y_f[:, 0:img.shape[1] - 1] = copy.copy(img[:, 1:])

    # x_backward
    x_b = copy.copy(img)
    x_b[1:, :] = copy.copy(img[0:img.shape[0] - 1, :])

    # y_backward
    y_b = copy.copy(img)
    y_b[:, 1:] = copy.copy(img[:, 0:img.shape[1] - 1])

    # x_forward_y_backward
    x_f_y_b = copy.copy(x_f)
    x_f_y_b[:, 1:] = copy.copy(x_f_y_b[:, 0:img.shape[1] - 1])

    # x_backward_y_forward
    x_b_y_f = copy.copy(x_b)
    x_b_y_f[:, 0:img.shape[1] - 1] = copy.copy(x_b_y_f[:, 1:])

    # x_mixed
    x_m = x_f_y_b - y_b
    # y_mixed
    y_m = x_b_y_f - x_b
    x_f -= img
    y_f -= img
    x_b = img - x_b
    y_b = img - y_b

    result = np.zeros(img.shape, np.double)
    epsilon_m = np.double(np.full((img.shape[0], img.shape[1]), epsilon))

    a1 = (x_f[:, :] + y_f[:, :]) / \
         np.maximum(epsilon_m, np.sqrt(x_f[:, :] * x_f[:, :] + y_f[:, :] * y_f[:, :]))
    a2 = x_b[:, :] / np.maximum(epsilon_m, np.sqrt(x_b[:, :] * x_b[:, :] + y_m[:, :] * y_m[:, :]))
    a3 = y_b[:, :] / np.maximum(epsilon_m, np.sqrt(x_m[:, :] * x_m[:, :] + y_b[:, :] * y_b[:, :]))
    result[:, :] = a1 - a2 - a3

    return result


def grad_tv(img, epsilon=1e-3):
    result = np.zeros(img.shape, np.double)
    for c in range(0, img.shape[2], 1):
        channel = copy.copy(img[:, :, c])
        result[:, :, c] = grad_tv_1c(channel, epsilon)
    return result