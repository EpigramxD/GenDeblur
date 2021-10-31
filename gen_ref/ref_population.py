from gen.individual import Individual
from gen.mutation import mutate
from utils.imgDeconv import ImgDeconv
from utils.imgQuality import ImgQuality
from utils.scalePyramid import ScalePyramidRef
from utils.size_utils import *

class PopulationRef:
    """
    Класс популяции
    """
    def __init__(self, sharp_img, blurred_img, min_kernel_size, step, max_kernel_size, no_ref_metric, ref_metric, deconv_type):
        """
        Конструктор
        :param sharp_img: четкое изображение
        :param blurred_img: размытое изображение
        :param max_kernel_size: максимальный размер ядра
        :param no_ref_metric: используемая для оценки качества не-референсная матрика (см. utils.metric.get_no_ref_quality)
        :param ref_metric: используемая для оценки качества референсная матрика (см. utils.metric.get_ref_qualiy)
        :param deconv_type: вид инверсной фильтрации
        """
        self.no_ref_metric = no_ref_metric
        self.ref_metric = ref_metric
        self.deconv_type = deconv_type
        # текущий размер ядра
        self.kernel_size = min_kernel_size
        self.scale_pyramid = ScalePyramidRef(sharp_img, blurred_img, min_kernel_size, step, max_kernel_size)
        self.sharp = self.scale_pyramid.images[self.kernel_size][0]
        self.blurred = self.scale_pyramid.images[self.kernel_size][1]
        # обновить размер популяции
        self.__update_pop_size()
        # создание особей
        self.individuals = [Individual(self.kernel_size, True) for i in range(0, self.size)]

    def fit(self):
        """
        Оценка приспособленностей особей популяции
        """
        for individual in self.individuals:
            deblurred_image = ImgDeconv.do_deconv(self.blurred, individual.psf, self.deconv_type)

            individual.score = ImgQuality.get_no_ref_quality(deblurred_image, self.no_ref_metric) + ImgQuality.get_ref_quality(self.sharp, deblurred_image, self.ref_metric)
            #individual.score = frob_metric2(deblurred_image, self.blurred, individual.psf)
            #individual.score = get_no_ref_quality(deblurred_image, self.no_ref_metric) #+ get_ref_qualiy(self.sharp, deblurred_image, self.ref_metric)
        self.individuals.sort(key=lambda x: x.score, reverse=True)

    def __update_pop_size(self, multiplier=10):
        """
        Обновление размера популяции
        :param multiplier: множитель (во сколько раз должна увеличиться популяция)
        :return:
        """
        self.size = int(self.kernel_size * multiplier)
        print("POPULATION SIZE UPDATED")

    def __expand_population(self):
        """
        Расширение популяции до указанного размера self.size
        """
        new_kernel_size_index = self.scale_pyramid.kernel_sizes.index(self.kernel_size) + 1
        new_kernel_size = self.scale_pyramid.kernel_sizes[new_kernel_size_index]
        self.kernel_size = new_kernel_size
        old_size = self.size
        self.__update_pop_size()
        copy_diff = copy.deepcopy(self.individuals[old_size - (self.size - old_size) - 1:old_size - 1])
        copy_diff = mutate(copy_diff, probability=0.1, add_prob=0.5, use_smart=True)
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
        self.sharp = copy.deepcopy(self.scale_pyramid.images[self.kernel_size][0])
        self.blurred = copy.deepcopy(self.scale_pyramid.images[self.kernel_size][1])
        # апскейльнуть каждую особь
        for individual in self.individuals:
            if upscale_type == "pad":
                individual.upscale_pad(self.kernel_size)
            elif upscale_type == "fill":
                individual.upscale_fill(self.kernel_size)

        print(f"POPULATION UPSCALED TO SIZE: {self.kernel_size}")

    def display(self):
        count_in_row = int(self.size / 10)
        count_in_col = 10

        width = count_in_row * self.kernel_size
        height = count_in_col * self.kernel_size

        mat = np.zeros((height, width), np.float32)

        i = 0
        j = 0
        for ind in self.individuals:
            mat[i * self.kernel_size:i * self.kernel_size + self.kernel_size, j * self.kernel_size:j * self.kernel_size + self.kernel_size] = copy.deepcopy(ind.psf)

            j += 1
            if j == count_in_row:
                j = 0
                i += 1

            if i == count_in_col:
                cv.imshow("all kernels", cv.resize(mat, None, fx=10, fy=10, interpolation=cv.INTER_AREA))