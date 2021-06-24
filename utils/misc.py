import copy
import cv2 as cv


def check_and_convert_to_grayscale(image):
    """
    Конвертация в черн-белое, если изображение цветное
    :param image: изображение
    :return: ЧБ изображение
    """
    if len(image.shape) == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        return image


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