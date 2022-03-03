import copy

import cv2 as cv
import time
import multiprocessing as mp
import numpy as np

from gen_ref.ref_population import PopulationRef
from utils.imgDeconv import ImgDeconv
from utils.imgQuality import ImgQuality
from utils.imgUtils import ImgUtils
from utils.scalePyramid import ScalePyramidRef
from .crossoverOperators import CrossoverOperators
from .selectionOperators import SelectionOperators
from .mutationOperators import MutationOperators


# TODO: ОТРЕФАКТОРИТЬ
def fit_range(blurred, deconv_type, no_ref_metric_type, population_range, empty_list):
    for individual in population_range:
        deblurred_image = ImgDeconv.do_deconv(blurred, individual.psf, deconv_type)
        # individual.score = ImgQuality.get_no_ref_quality(deblurred_image, no_ref_metric_type) + ImgQuality.get_ref_quality(self.__sharp, deblurred_image, ref_metric_type)
        # individual.score = ImgQuality.frob_metric_simple(deblurred_image, self.__blurred, individual.psf) + 100000 * ImgQuality.get_no_ref_quality(deblurred_image, no_ref_metric_type)
        individual.score = ImgQuality.test_map_metric(deblurred_image, blurred) + 5000 * ImgQuality.get_no_ref_quality(deblurred_image, no_ref_metric_type)
        #individual.score = ImgQuality.frob_metric2(deblurred_image, blurred, individual.psf) + ImgQuality.get_no_ref_quality(deblurred_image, no_ref_metric_type)
        # individual.score = get_no_ref_quality(deblurred_image, self.no_ref_metric) #+ get_ref_qualiy(self.__sharp, deblurred_image, self.ref_metric)
        empty_list.append(copy.deepcopy(individual))


class GenDeblurrer(object):
    def __init__(self, stagnation_pop_count,
                 ref_metric_type,
                 no_ref_metric_type,
                 deconv_type,
                 selection_args,
                 crossover_args,
                 mutation_args,
                 pyramid_args,
                 population_expand_factor,
                 elite_count,
                 upscale_type,
                 multiprocessing_manager):

        self.__stagnation_pop_count = stagnation_pop_count
        # параметры селекции
        self.__selection_args = selection_args,
        # параметры скрещивания
        self.__crossover_args = crossover_args,
        # парамеры мутации
        self.__mutation_args = mutation_args,
        self.__elite_count = elite_count
        self.__ref_metric_type = ref_metric_type
        self.__no_ref_metric_type = no_ref_metric_type
        self.__deconv_type = deconv_type
        self.__pyramid_args = pyramid_args
        self.__population_expand_factor = population_expand_factor
        self.__upscale_type = upscale_type
        self.__multiprocessing_manager = multiprocessing_manager

    def deblur(self, sharp_img, blurred_img):
        sharp_img_gray = ImgUtils.to_grayscale(sharp_img)
        sharp_img_gray = ImgUtils.im2double(sharp_img_gray)

        blurred_img_gray = ImgUtils.to_grayscale(blurred_img)
        blurred_img_gray = ImgUtils.im2double(blurred_img_gray)

        scale_pyramid = ScalePyramidRef(sharp_img_gray, blurred_img_gray,
                                        self.__pyramid_args["min_psf_size"],
                                        self.__pyramid_args["step"],
                                        self.__pyramid_args["max_psf_size"],
                                        self.__pyramid_args["inter_type"])
        self.__population = PopulationRef(scale_pyramid, expand_factor=self.__population_expand_factor)

        best_quality_in_pop = -10000000000.0
        upscale_flag = 0
        best_ever_kernel = np.zeros(sharp_img_gray.shape)
        best_kernels = []

        for i in range(0, len(scale_pyramid.psf_sizes), 1):
            while True:
                if upscale_flag == self.__stagnation_pop_count:
                    upscale_flag = 0
                    break
                start = time.time()

                if self.__population.current_psf_size < 13:
                    self.__population.fit(self.__no_ref_metric_type, self.__ref_metric_type, self.__deconv_type)
                else:
                    # TODO: отрефакторить
                    # параллельная оценка приспособленности популяции по частям
                    population_size = len(self.__population.individuals)
                    population_step = int((population_size - (population_size % 4)) / 4)
                    fitted_population_mp = self.__multiprocessing_manager.list()

                    p1 = mp.Process(target=fit_range, args=(copy.deepcopy(self.__population.blurred), self.__deconv_type, self.__no_ref_metric_type, self.__population.individuals[0: population_step], fitted_population_mp))
                    p2 = mp.Process(target=fit_range, args=(copy.deepcopy(self.__population.blurred), self.__deconv_type, self.__no_ref_metric_type, self.__population.individuals[population_step: population_step * 2], fitted_population_mp))
                    p3 = mp.Process(target=fit_range, args=(copy.deepcopy(self.__population.blurred), self.__deconv_type, self.__no_ref_metric_type, self.__population.individuals[population_step * 2: population_step * 3], fitted_population_mp))
                    p4 = mp.Process(target=fit_range, args=(copy.deepcopy(self.__population.blurred), self.__deconv_type, self.__no_ref_metric_type, self.__population.individuals[population_step * 3: population_size], fitted_population_mp))

                    p1.start()
                    p2.start()
                    p3.start()
                    p4.start()

                    p1.join()
                    p2.join()
                    p3.join()
                    p4.join()
                    fitted_population = copy.deepcopy(fitted_population_mp)
                    fitted_population.sort(key=lambda x: x.score, reverse=True)
                    self.__population.individuals = copy.deepcopy(fitted_population)

                end = time.time()
                print(f"ELAPSED TIME: {end-start}")
                print(f"best quality in pop: {self.__population.individuals[0].score}, best quality ever: {best_quality_in_pop}")

                if self.__population.individuals[0].score > best_quality_in_pop:
                    best_quality_in_pop = copy.deepcopy(self.__population.individuals[0].score)
                    best_ever_kernel = copy.deepcopy(self.__population.individuals[0].psf)
                    best_kernels.append(copy.deepcopy(self.__population.individuals[0].psf))
                    upscale_flag = 0

                # селекция
                elite_individuals, non_elite_individuals = self.__population.get_elite_non_elite(self.__elite_count)
                self.__selection_args[0]["k"] = len(non_elite_individuals)
                selected_individuals = SelectionOperators.select(non_elite_individuals, self.__selection_args[0])
                # скрещивание
                crossed_individuals = CrossoverOperators.crossover(selected_individuals, self.__crossover_args[0])
                # мутация
                mutated_individuals = MutationOperators.mutate(crossed_individuals, self.__mutation_args[0])
                # TODO: рассмотреть вариант с динамической положительной вероятностью мутации
                # self.__mutation_args[0]["pos_probability"] = 2.7 * 1 / scale_pyramid.psf_sizes[i]

                # обновление особей популяции
                self.__population.individuals.clear()
                self.__population.individuals.extend(copy.deepcopy(mutated_individuals))
                self.__population.individuals.extend(copy.deepcopy(elite_individuals))
                upscale_flag += 1

            # апскейлим
            if i != len(scale_pyramid.psf_sizes) - 1:
                # xd_test = np.zeros((population.kernel_size, population.kernel_size), np.float32)
                # for kernel in best_kernels:
                #     xd_test += kernel
                # xd_test += population.individuals[0].psf
                # cv.normalize(xd_test, xd_test, 0.0, 1.0, cv.NORM_MINMAX)
                # best_kernels.clear()
                # cv.imshow("xd_test", cv.resize(xd_test, None, fx=10, fy=10, interpolation=cv.INTER_AREA))

                # ЗАПИСЬ
                best_result_for_size = ImgDeconv.do_deconv(self.__population.blurred, self.__population.individuals[0].psf, type=self.__deconv_type)
                cv.normalize(best_result_for_size, best_result_for_size, 0.0, 255.0, cv.NORM_MINMAX)

                lel_test = copy.deepcopy(self.__population.individuals[0].psf)
                best_kernel_for_size = cv.resize(lel_test, None, fx=10, fy=10, interpolation=cv.INTER_AREA)
                cv.normalize(best_kernel_for_size, best_kernel_for_size, 0, 255, cv.NORM_MINMAX)
                # cv.imshow("best_kernel_for_size", best_kernel_for_size)

                blurred_normalized = copy.deepcopy(self.__population.blurred)
                cv.normalize(blurred_normalized, blurred_normalized, 0, 255, cv.NORM_MINMAX)

                result_file_name = "../images/results/restored_size_{}.jpg".format(self.__population.current_psf_size)
                blurred_file_name = "../images/results/blurred_size_{}.jpg".format(self.__population.current_psf_size)
                kernel_file_name = "../images/results/kernel_size_{}.jpg".format(self.__population.current_psf_size)

                cv.imwrite(result_file_name, best_result_for_size)
                cv.imwrite(kernel_file_name, best_kernel_for_size)
                cv.imwrite(blurred_file_name, blurred_normalized)

                best_quality_in_pop = -10000000000.0
                self.__population.upscale(self.__upscale_type)

            if i == len(scale_pyramid.psf_sizes) - 1:
                # xd_test = np.zeros((population.kernel_size, population.kernel_size), np.float32)
                # for kernel in best_kernels:
                #     xd_test += kernel
                # xd_test += population.individuals[0].psf
                # cv.normalize(xd_test, xd_test, 0.0, 1.0, cv.NORM_MINMAX)
                # best_kernels.clear()
                # cv.imshow("xd_test", cv.resize(xd_test, None, fx=10, fy=10, interpolation=cv.INTER_AREA))

                # ЗАПИСЬ
                self.__population.fit(self.__no_ref_metric_type, self.__ref_metric_type, self.__deconv_type)
                best_result_for_size = ImgDeconv.do_deconv(self.__population.blurred, self.__population.individuals[0].psf, type=self.__deconv_type)
                cv.normalize(best_result_for_size, best_result_for_size, 0.0, 255.0, cv.NORM_MINMAX)

                lel_test = copy.deepcopy(self.__population.individuals[0].psf)
                best_kernel_for_size = cv.resize(lel_test, None, fx=10, fy=10, interpolation=cv.INTER_AREA)
                cv.normalize(best_kernel_for_size, best_kernel_for_size, 0, 255, cv.NORM_MINMAX)
                # cv.imshow("best_kernel_for_size", best_kernel_for_size)

                blurred_normalized = copy.deepcopy(self.__population.blurred)
                cv.normalize(blurred_normalized, blurred_normalized, 0, 255, cv.NORM_MINMAX)

                result_file_name = "../images/results/restored_size_{}.jpg".format(self.__population.current_psf_size)
                blurred_file_name = "../images/results/blurred_size_{}.jpg".format(self.__population.current_psf_size)
                kernel_file_name = "../images/results/kernel_size_{}.jpg".format(self.__population.current_psf_size)

                cv.imwrite(result_file_name, best_result_for_size)
                cv.imwrite(kernel_file_name, best_kernel_for_size)
                cv.imwrite(blurred_file_name, blurred_normalized)
                # ЗАПИСЬ

        return ImgDeconv.do_deconv(self.__population.blurred, self.__population.individuals[0].psf, self.__deconv_type, K=1.0)