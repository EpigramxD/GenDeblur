from gen.individual import Individual
from utils.imgDeconv import ImgDeconv
from utils.imgQuality import ImgQuality
from utils.scalePyramid import ScalePyramidRef
from utils.size_utils import *

class PopulationRef:
    """
    Класс популяции
    """
    def __init__(self, sharp_img, blurred_img, min_kernel_size, step, max_kernel_size):
        """
        Конструктор
        :param sharp_img: четкое изображение
        :param blurred_img: размытое изображение
        :param max_kernel_size: максимальный размер ядра
        :param deconv_type: вид инверсной фильтрации
        """
        # текущий размер ядра
        self.__current_psf_size = min_kernel_size
        self.__scale_pyramid = ScalePyramidRef(sharp_img, blurred_img, min_kernel_size, step, max_kernel_size)
        self.__sharp = self.__scale_pyramid.images[self.__current_psf_size][0]
        self.__blurred = self.__scale_pyramid.images[self.__current_psf_size][1]
        # обновить размер популяции
        self.__update_pop_size()
        # создание особей
        self.__individuals = [Individual(self.__current_psf_size, True) for i in range(0, self.size)]

    @property
    def current_psf_size(self):
        return self.__current_psf_size

    @property
    def scale_pyramid(self):
        return self.__scale_pyramid

    @property
    def sharp(self):
        return self.__sharp

    @property
    def blurred(self):
        return self.__blurred

    @property
    def individuals(self):
        return self.__individuals


    def fit(self, no_ref_metric_type, ref_metric_type, deconv_type):
        """
        Оценка приспособленностей особей популяции
        """
        for individual in self.__individuals:
            deblurred_image = ImgDeconv.do_deconv(self.__blurred, individual.psf, deconv_type)

            individual.score = ImgQuality.get_no_ref_quality(deblurred_image, no_ref_metric_type) + ImgQuality.get_ref_quality(self.__sharp, deblurred_image, ref_metric_type)
            #individual.score = frob_metric2(deblurred_image, self.__blurred, individual.psf)
            #individual.score = get_no_ref_quality(deblurred_image, self.no_ref_metric) #+ get_ref_qualiy(self.__sharp, deblurred_image, self.ref_metric)
        self.__individuals.sort(key=lambda x: x.score, reverse=True)

    def get_elite_non_elite(self, elite_count):
        """
        Получить элитных и не элитных особей
        :param elite_count: количество элитных особей
        :return: (элитные, не элитные)
        """
        return copy.deepcopy(self.__individuals[:elite_count]), copy.deepcopy(self.__individuals[elite_count:])

    def __update_pop_size(self, multiplier=10):
        """
        Обновление размера популяции
        :param multiplier: множитель (во сколько раз должна увеличиться популяция)
        :return:
        """
        self.size = int(self.__current_psf_size * multiplier)
        print("POPULATION SIZE UPDATED")

    def __expand_population(self):
        """
        Расширение популяции до указанного размера self.size
        """
        new_kernel_size_index = self.__scale_pyramid.kernel_sizes.index(self.__current_psf_size) + 1
        new_kernel_size = self.__scale_pyramid.kernel_sizes[new_kernel_size_index]
        self.__current_psf_size = new_kernel_size
        old_size = self.size
        self.__update_pop_size()
        copy_diff = copy.deepcopy(self.__individuals[old_size - (self.size - old_size) - 1:old_size - 1])
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
        self.__sharp = copy.deepcopy(self.__scale_pyramid.images[self.__current_psf_size][0])
        self.__blurred = copy.deepcopy(self.__scale_pyramid.images[self.__current_psf_size][1])
        # апскейльнуть каждую особь
        for individual in self.__individuals:
            if upscale_type == "pad":
                individual.upscale_pad(self.__current_psf_size)
            elif upscale_type == "fill":
                individual.upscale_fill(self.__current_psf_size)

        print(f"POPULATION UPSCALED TO SIZE: {self.__current_psf_size}")

    def display(self):
        count_in_row = int(self.size / 10)
        count_in_col = 10

        width = count_in_row * self.__current_psf_size
        height = count_in_col * self.__current_psf_size

        mat = np.zeros((height, width), np.float32)

        i = 0
        j = 0
        for ind in self.__individuals:
            mat[i * self.__current_psf_size:i * self.__current_psf_size + self.__current_psf_size, j * self.__current_psf_size:j * self.__current_psf_size + self.__current_psf_size] = copy.deepcopy(ind.psf)

            j += 1
            if j == count_in_row:
                j = 0
                i += 1

            if i == count_in_col:
                cv.imshow("all kernels", cv.resize(mat, None, fx=10, fy=10, interpolation=cv.INTER_AREA))