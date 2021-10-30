import cv2 as cv
from .filteringUtils import FilteringUtils

class ScalePyramidRef(object):
    """
    Класс пирамиды четких и размытых изображений разного разрешения
    self.sizes - список размеров ядра (разрешения функции искажения)
    self.images - мапа. Ключ - разрешение ядра. Значение - tuple (четкое изображение, размытое изображение)
    """

    def __init__(self, sharp_img, blurred_img, min_kernel_size=3, step=2, max_kernel_size=23, inter_type=cv.INTER_AREA):
        """
        Конструктор
        :param sharp_img: четкое изображение
        :param blurred_img: размытое изображение
        :param min_kernel_size: минимальное разрешение ядра
        :param step: приращение разрешения ядра при переходе на новый уровень пирамиды
        :param max_kernel_size: максимальный размер ядра
        :param inter_type: вид интерполяции
        """
        self.__sizes = self.__get_sizes(min_kernel_size, step, max_kernel_size)
        self.__images = dict()
        self.__build(sharp_img, blurred_img, max_kernel_size, inter_type)

    @property
    def sizes(self):
        """
        Получить все размеры ядер
        :return:
        """
        return self.__sizes

    @property
    def images(self):
        """
        Получить мапу с изображениями
        :return: мапа
        """
        return self.__images

    def __build(self, sharp_img, blurred_img, min_kernel_size, inter_type):
        """
        Построить пирамиду
        :param sharp_img: четкое изображение
        :param blurred_img: искаженное изображение
        :param min_kernel_size: максимальный размер ядра (разрешение функции искажения)
        :param inter_type: вид интерполяции
        """
        for size in self.sizes:
            multiplier = size / min_kernel_size
            if multiplier != 1:
                sharp_resized = cv.resize(sharp_img, None, fx=multiplier, fy=multiplier, interpolation=inter_type)
                blurred_resized = cv.resize(blurred_img, None, fx=multiplier, fy=multiplier, interpolation=inter_type)
            else:
                sharp_resized = sharp_img.copy()
                blurred_resized = blurred_img.copy()
            sharp_resized = FilteringUtils.im2double(sharp_resized)
            blurred_resized = FilteringUtils.im2double(blurred_resized)
            self.__images[size] = (sharp_resized, blurred_resized)

    def __get_sizes(self, min_kernel_size, step, max_kernel_size):
        """
        Получить размеры ядра
        :param min_kernel_size: минимальный размер ядра (разрешение функции искажения)
        :param step: приращение разрешения ядра при переходе на новый уровень пирамиды
        :param max_kernel_size: максимальный размер ядра (разрешение функции искажения)
        :return: список размеров ядер
        """
        kernel_sizes = []
        current_kernel_size = min_kernel_size
        while current_kernel_size <= max_kernel_size:
            kernel_sizes.append(current_kernel_size)
            current_kernel_size += step

        if kernel_sizes[len(kernel_sizes) - 1] < max_kernel_size:
            kernel_sizes.append(max_kernel_size)
        return kernel_sizes