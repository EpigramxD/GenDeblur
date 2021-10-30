import copy
import cv2 as cv
import numpy as np


class ImgUtils(object):
    @staticmethod
    def im2double(img):
        """
        Конвертировать изображение в float32 и нормализовать его
        :param img: изображение
        :return: float32 изображение
        """
        result = np.float32(img)
        cv.normalize(result, result, 0.0, 1.0, cv.NORM_MINMAX)
        return result

    @staticmethod
    def is1c(img):
        return len(img.shape) == 2 or img.shape[2] == 1

    @staticmethod
    def is3c(img):
        return len(img.shape) == 3 and img.shape[2] == 3

    @staticmethod
    def to_grayscale(img):
        """
        Конвертация в черн-белое, если изображение цветное
        :param img: изображение
        :return: ЧБ изображение
        """
        if ImgUtils.is3c(img) and not ImgUtils.is1c(img):
            return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            return img.copy()

    @staticmethod
    def pad_to_shape(img, shape):
        """
        Расширить изображение до нужной формы нулями (черными рамками по краям)
        :param img: расширяемое изображение
        :param shape: новая форма - tuple (высота, ширина)
        :return: рсширенное изображение
        """
        if shape[0] == img.shape[0] and shape[1] == img.shape[1]:
            return copy.deepcopy(img)

        elif shape[0] < img.shape[0] or shape[1] < img.shape[1]:
            raise AttributeError("Новый размер должен быть меньше размера изображения")

        center_y = int(shape[0] / 2)
        center_x = int(shape[1] / 2)

        half_height = int(img.shape[0] / 2)
        half_width = int(img.shape[1] / 2)

        x_index_from = center_x - half_width
        x_index_to = center_x + img.shape[1] - half_width

        y_index_from = center_y - half_height
        y_index_to = center_y + img.shape[0] - half_height

        result = np.zeros(shape, np.float32)
        result[y_index_from:y_index_to, x_index_from:x_index_to] = copy.deepcopy(img)
        cv.normalize(result, result, 0.0, 1.0, cv.NORM_MINMAX)
        return result

    @staticmethod
    def get_dft(img, dft_type=cv.DFT_COMPLEX_OUTPUT):
        """
        Получить образ Фурье изображения
        :param img: изображение
        :param dft_type: вид DFT
        :return: DFT
        """
        planes = [np.float32(img), np.zeros(img.shape, np.float32)]
        dft_ready = cv.merge(planes)
        cv.dft(dft_ready, dft_ready, flags=dft_type)
        return dft_ready

    @staticmethod
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

    @staticmethod
    def __freq_filter_1c(img, filter):
        """
        Частотная фильтрация для одного канала
        :param img: фильтруемое изображение (float32, [0, 1])
        :param filter: ядро свертки (float32, [0, 1])
        :return: результат фильтрации (float32, [0, 1])
        """
        if ImgUtils.is1c(img):
            kernel = ImgUtils.pad_to_shape(filter, img.shape)
            image_copy = np.copy(img)
            image_fft = np.fft.fft2(image_copy)
            filter_fft = np.fft.fft2(kernel)
            fft_mul = image_fft * filter_fft
            fft_mul = np.abs(np.fft.ifft2(fft_mul))
            result = np.zeros(fft_mul.shape)
            result = np.float32(result)
            fft_mul = np.float32(fft_mul)
            cv.normalize(fft_mul, result, 0.0, 1.0, cv.NORM_MINMAX)
            return np.fft.fftshift(result)
        else:
            raise AttributeError("Wrong image format, image should have 1 channel")

    @staticmethod
    def freq_filter(img, filter):
        """
        Частотная фильтрация для любого количества каналов
        :param img: фильтруемое изображение (float32, [0, 1])
        :param filter: ядро свертки (float32, [0, 1])
        :return: результат фильтрации (float32, [0, 1])
        """
        channels = cv.split(img)
        for i in range(0, len(channels), 1):
            channels[i] = ImgUtils.__freq_filter_1c(channels[i], filter)
        result = cv.merge(channels)
        return ImgUtils.im2double(result)

    @staticmethod
    def pad_to_shape(img, shape):
        """
        Расширить изображение до нужной формы нулями (черными рамками по краям)
        :param img: расширяемое изображение
        :param shape: новая форма - tuple (высота, ширина)
        :return: рсширенное изображение
        """
        if shape[0] == img.shape[0] and shape[1] == img.shape[1]:
            return copy.deepcopy(img)

        elif shape[0] < img.shape[0] or shape[1] < img.shape[1]:
            raise AttributeError("Новый размер должен быть меньше размера изображения")

        center_y = int(shape[0] / 2)
        center_x = int(shape[1] / 2)

        half_height = int(img.shape[0] / 2)
        half_width = int(img.shape[1] / 2)

        x_index_from = center_x - half_width
        x_index_to = center_x + img.shape[1] - half_width

        y_index_from = center_y - half_height
        y_index_to = center_y + img.shape[0] - half_height

        result = np.zeros(shape, np.float32)
        result[y_index_from:y_index_to, x_index_from:x_index_to] = copy.deepcopy(img)
        cv.normalize(result, result, 0.0, 1.0, cv.NORM_MINMAX)
        return result
