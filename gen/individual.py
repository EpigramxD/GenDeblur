import numpy as np
import cv2 as cv
import copy
import random


class Individual:
    """
    Класс, представляющий особь
    """
    def __init__(self, *args):
        """
        Конструктор
        :param args: количество парметров конструктора
        """
        # Если параметр 1, то это psf, устанавливаем его и устанавливаем размеры
        if len(args) == 1:
            if isinstance(args[0], np.ndarray):
                self.psf = args[0].copy()
                if self.psf.shape[0] != self.psf.shape[1]:
                    raise AttributeError("Ядро смаза должно иметь одинаковую высоту и ширину".format(type(args[0])))
                self.kernel_size = self.psf.shape[0]
                self.score = 0.0
            else:
                raise AttributeError("Получен аргумент типа {}, а должен быть ndarray".format(type(args[0])))

        # Если параметров 2, то это начальный размер psf и boolean 'это линия'
        elif len(args) == 2:
            kernel_size = args[0]
            is_line = args[1]
            if isinstance(kernel_size, int) and isinstance(is_line, bool):
                self.kernel_size = kernel_size
                self.score = 0.0
                self.psf = np.zeros((kernel_size, kernel_size))
                if is_line:
                    self.psf[1][0] = 1.0
                    self.psf[1][1] = 1.0
                    self.psf[1][2] = 1.0
            else:
                raise AttributeError("Получены аргументы типа {} и {}, а должны быть int и bool".format(type(args[0]), type(args[1])))

        self.normalize()

    def normalize(self):
        """
        Нормализация psf
        """
        self.psf = np.float32(self.psf)
        cv.normalize(self.psf, self.psf, 0.0, 1.0, cv.NORM_MINMAX)

    def mutate(self, probability=0.05, add_prob=0.000001):
        """
        Мутация особи
        :param probability: вероятность мутирования
        :param add_prob: вероятность добавления рандомного значения к гену, если меньше, то вычитание
        """
        for row in range(0, self.kernel_size):
            for col in range(0, self.kernel_size):
                if random.random() < probability:
                    if random.random() < add_prob:
                        self.psf[row][col] += random.random()
                        if self.psf[row][col] > 1.0:
                            self.psf[row][col] = 1.0
                    else:
                        self.psf[row][col] -= random.random()
                        if self.psf[row][col] < 0.0:
                            self.psf[row][col] = 0
        self.normalize()

    def upscale(self, new_kernel_size):
        """
        Увеличение размера ядра путем растягивания (масштабирования)
        :param new_kernel_size: новый размер ядра
        """
        multiplier = new_kernel_size / self.kernel_size
        self.kernel_size = new_kernel_size
        self.psf = copy.deepcopy(cv.resize(self.psf, None, fx=multiplier, fy=multiplier, interpolation=cv.INTER_CUBIC))
        self.normalize()

    def upscale_pad(self, new_kernel_size):
        """
        Увеличение размера ядра путем добавления нулей по краям
        :param new_kernel_size: новый размер ядра
        """
        result = np.zeros((new_kernel_size, new_kernel_size), np.float32)
        difference = new_kernel_size - self.psf.shape[0]
        start_pos = int(difference/2)
        result[start_pos:start_pos + self.psf.shape[0], start_pos:start_pos + self.kernel_size] = copy.deepcopy(self.psf)
        self.psf = copy.deepcopy(result)
        self.kernel_size = new_kernel_size
        self.normalize()
        return result