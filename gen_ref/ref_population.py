from gen.individual import Individual
from gen.mutation import mutate
from utils.size_utils import *
from utils.metric import get_quality
from utils.deconv import do_RL_deconv, do_weiner_deconv_1c
from skimage.metrics import peak_signal_noise_ratio
from utils.misc import *

# TODO по-человечески потом отнаследоваться от класса Population
class RefPopulation:
    """
    Класс популяции
    """
    def __init__(self, sharp, blurred, max_kernel_size, metric_type):
        """
        Конструктор
        :param sharp: четкое изображение
        :param blurred: размытое изображение
        :param max_kernel_size: максимальный размер ядра
        :param metric_type: используемая для оценки качества матрика (см. utils.metric.get_quality)
        """
        self.metric_type = metric_type
        # начальный размер ядра
        self.kernel_size = 3
        # получить пирамиду размеров
        self.size_pyramid = build_double_size_pyramid(sharp, blurred, max_kernel_size)
        # установить начальное размытое изображение самого маленького размера в ядре
        self.sharp = self.size_pyramid[self.kernel_size][0]
        self.blurred = self.size_pyramid[self.kernel_size][1]
        # список всех размеров ядер
        self.kernel_sizes = list(self.size_pyramid)
        # обновить размер популяции
        self.__update_pop_size()
        # создание особей
        self.individuals = [Individual(self.kernel_size, self.kernel_size==3) for i in range(0, self.size)]
        self.individuals[0].psf[0][2] = 1.0
        self.individuals[0].psf[1][1] = 1.0
        self.individuals[0].psf[2][0] = 1.0

    # оценка приспособленностей особи
    def fit(self, deconvolution_type):
        """
        Оценка приспособленностей особи
        :param deconvolution_type: вид инверсной фильтрации (деконволюции): "weiner" - фильтр винера, "LR" - метод Люси-Ричардсона
        """
        for individual in self.individuals:
            if deconvolution_type == "weiner":
                deblurred_image = do_weiner_deconv_1c(self.blurred, individual.psf, 0.0001)
            elif deconvolution_type == "LR":
                deblurred_image = do_RL_deconv(self.blurred, individual.psf, iterations=10)
            individual.score = get_quality(deblurred_image, self.metric_type) * peak_signal_noise_ratio(self.sharp, deblurred_image)
        self.individuals.sort(key=lambda x: x.score, reverse=True)

    def __update_pop_size(self, multiplier=10):
        """
        Обновление размера популяции
        :param multiplier: множитель (во сколько раз должна увеличиться популяция)
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
        self.sharp = copy.deepcopy(self.size_pyramid[self.kernel_size][0])
        self.blurred = copy.deepcopy(self.size_pyramid[self.kernel_size][1])
        # апскейльнуть каждую особь
        for individual in self.individuals:
            if upscale_type == "pad":
                individual.upscale_pad(self.kernel_size)
            elif upscale_type == "fill":
                individual.upscale(self.kernel_size)

        print("POPULATION UPSCALED")

