import copy

from gen.individual import Individual
from utils.imgDeconv import ImgDeconv
from utils.imgQuality import ImgQuality


class Population:
    """
    Класс популяции
    """
    def __init__(self, scale_pyramid, population_size=10):
        """
        Конструктор
        :param scale_pyramid: scale-пирамида
        :param population_size: количество особей в популяции
        """
        # текущий размер ядра
        self.__scale_pyramid = scale_pyramid
        self.__size = population_size
        self.__minimum_size = population_size
        self.__current_psf_size = scale_pyramid.psf_sizes[0]
        self.__blurred = self.__scale_pyramid.images[self.__current_psf_size]
        # обновить размер популяции
        self.__update_pop_size()
        # создание особей
        self.__individuals = [Individual(self.__current_psf_size, True) for i in range(0, self.__size)]

    @property
    def current_psf_size(self):
        return self.__current_psf_size

    @property
    def scale_pyramid(self):
        return self.__scale_pyramid

    @property
    def blurred(self):
        return self.__blurred

    @property
    def individuals(self):
        return self.__individuals

    @individuals.setter
    def individuals(self, individuals):
        self.__individuals = copy.deepcopy(individuals)

    @property
    def expand_factor(self):
        return self.expand_factor

    @property
    def size(self):
        return self.__size


    def fit(self, no_ref_metric_type, ref_metric_type, deconv_type):
        """
        Оценка приспособленностей особей популяции
        """
        for individual in self.__individuals:
            deblurred_image = ImgDeconv.do_deconv(self.__blurred, individual.psf, deconv_type)
            individual.score = ImgQuality.test_map_metric(deblurred_image, self.__blurred)
            individual.score += ImgQuality.get_no_ref_quality(deblurred_image, no_ref_metric_type)

        self.__individuals.sort(key=lambda x: x.score, reverse=True)


    def get_elite_non_elite(self, elite_count):
        """
        Получить элитных и не элитных особей
        :param elite_count: количество элитных особей
        :return: (элитные, не элитные)
        """
        return copy.deepcopy(self.__individuals[:elite_count]), copy.deepcopy(self.__individuals[elite_count:])

    def __update_pop_size(self):
        """
        Обновление размера популяции
        :return:
        """
        self.__size = self.__minimum_size + self.__current_psf_size
        print("POPULATION SIZE UPDATED")

    def __expand_population(self):
        """
        Расширение популяции до указанного размера self.__size
        """
        new_kernel_size_index = self.__scale_pyramid.psf_sizes.index(self.__current_psf_size) + 1
        new_kernel_size = self.__scale_pyramid.psf_sizes[new_kernel_size_index]
        self.__current_psf_size = new_kernel_size
        old_size = self.__size
        self.__update_pop_size()
        copy_diff = copy.deepcopy(self.__individuals[old_size - (self.__size - old_size) - 1:old_size - 1])
        # мутация, чтобы популяция была более разнообразной
        for ind in copy_diff:
            ind.mutate_smart(probability=0.1, pos_prob=0.5)
        self.__individuals.extend(copy.deepcopy(copy_diff))
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
        self.__blurred = copy.deepcopy(self.__scale_pyramid.images[self.__current_psf_size])
        # апскейльнуть каждую особь
        for individual in self.__individuals:
            if upscale_type == "pad":
                individual.upscale_pad(self.__current_psf_size)
            elif upscale_type == "fill":
                individual.upscale_fill(self.__current_psf_size)

        print(f"POPULATION UPSCALED TO SIZE: {self.__current_psf_size}")