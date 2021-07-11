from gen.individual import Individual
from gen.mutation import mutate
from utils.size_utils import *
from utils.deconv import do_RL_deconv, do_wiener_deconv_1c
from utils.metric import get_no_ref_quality
from utils.misc import *


class Population:
    """
    Класс популяции
    """
    def __init__(self, image, max_kernel_size, metric_type):
        """
        Конструктор
        :param image: размытое изображение
        :param max_kernel_size: максимальный размер ядра
        :param metric_type: используемая для оценки качества матрика (см. utils.metric.get_quality)
        """
        self.metric_type = metric_type
        # начальный размер ядра, всегда начинаем с ядра 3x3
        self.kernel_size = 3
        # получить пирамиду размеров
        self.size_pyramid = build_size_pyramid(image, max_kernel_size)
        # установить начальное размытое изображение самого маленького размера в ядре
        self.image = self.size_pyramid[self.kernel_size]
        # список всех размеров ядер
        self.kernel_sizes = list(self.size_pyramid)
        # обновить размер популяции
        self.__update_pop_size()
        # создание особей
        self.individuals = [Individual(self.kernel_size, self.kernel_size==3) for i in range(0, self.size)]
        self.individuals[0].psf[0][2] = 1.0
        self.individuals[0].psf[1][1] = 1.0
        self.individuals[0].psf[2][0] = 1.0

    def fit(self, deconvolution_type):
        """
        Оценка приспособленностей особей
        :param deconvolution_type: вид инверсной фильтрации (деконволюции): "weiner" - фильтр винера, "LR" - метод Люси-Ричардсона
        """
        for individual in self.individuals:
            if deconvolution_type == "weiner":
                deblurred_image = do_wiener_deconv_1c(self.image, individual.psf, 1000)
            elif deconvolution_type == "LR":
                deblurred_image = do_RL_deconv(self.image, individual.psf, iterations=1)
            # обрезка краев, чтобы не портили оценку
            #deblurred_image = crop_image(deblurred_image, int(deblurred_image.shape[1]/10), int(deblurred_image.shape[0]/10))
            individual.score = get_no_ref_quality(deblurred_image, self.metric_type) #- np.count_nonzero(individual.psf)
        self.individuals.sort(key=lambda x: x.score, reverse=True)

    def __update_pop_size(self, multiplier=10):
        """
        Обновление размера популяции
        :param multiplier: множитель (во сколько раз должна увеличиться популяция, чем больше размер ядра, тем больше будет популяция)
        :return:
        """
        self.size = self.kernel_size * multiplier
        print("POPULATION SIZE UPDATED")

    def __expand_population(self):
        """
        Расширение популяции до указанного размера self.size
        :return:
        """
        new_kernel_size_index = self.kernel_sizes.index(self.kernel_size) + 1
        new_kernel_size = self.kernel_sizes[new_kernel_size_index]
        self.kernel_size = new_kernel_size
        old_size = self.size
        self.__update_pop_size()
        copy_diff = copy.deepcopy(self.individuals[old_size - (self.size - old_size) - 1:old_size - 1])
        copy_diff = mutate(copy_diff)
        self.individuals.extend(copy.deepcopy(copy_diff))
        print("POPULATION EXPANDED")

    def upscale(self, upscale_type):
        """
        Выполнить upscale всей популяции
        :param upscale_type: тип upscale ("pad" - заполняет недостающее нулями, "fill" - растягивание до размера)
        :return:
        """
        # расширение популяции
        self.__expand_population()
        # получить следующее по размеру размытое изображение из пирамиды
        self.image = copy.deepcopy(self.size_pyramid[self.kernel_size])

        # апскейльнуть каждую особь
        if upscale_type == "pad":
            for individual in self.individuals:
                individual.upscale_pad(self.kernel_size)

        elif upscale_type == "fill":
            for individual in self.individuals:
                individual.upscale_fill(self.kernel_size)
        print("POPULATION UPSCALED")

