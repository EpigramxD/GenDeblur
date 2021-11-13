import cv2 as cv
import numpy as np
import copy

import utils.imgDeconv
from gen_ref.ref_population import PopulationRef
from utils.imgDeconv import ImgDeconv
from .selectionOperators import SelectionOperators
from .crossoverOperators import CrossoverOperators

class GenDeblurrer(object):
    def __init__(self, stagnation_pop_count, ref_metric_type, no_ref_metric_type, deconv_type, crossover_prob, mutation_prob, pos_prob, min_psf_size, step, max_psf_size, elite_count):
        self.__stagnation_pop_count = stagnation_pop_count
        self.__ref_metric_type = ref_metric_type
        self.__no_ref_metric_type = no_ref_metric_type
        self.__deconv_type = deconv_type
        self.__crossover_prob = crossover_prob
        self.__mutation_prob = mutation_prob
        self.__pos_prob = pos_prob
        self.__min_psf_size = min_psf_size
        self.__step = step
        self.__max_psf_size = max_psf_size
        self.__elite_count = elite_count

    def deblur(self, sharp_img, blurred_img):
        self.__population = PopulationRef(sharp_img, blurred_img, self.__min_psf_size, self.__step, self.__max_psf_size)
        best_quality_in_pop = -10000.0
        upscale_flag = 0
        best_ever_kernel = np.zeros(sharp_img.shape)
        best_kernels = []

        for i in range(0, len(self.__population.scale_pyramid.psf_sizes), 1):
            while True:
                if upscale_flag == self.__stagnation_pop_count:
                    upscale_flag = 0
                    break

                self.__population.fit(self.__no_ref_metric_type, self.__ref_metric_type, self.__deconv_type)

                print(f"best quality in pop: {self.__population.individuals[0].score}, best quality ever: {best_quality_in_pop}")

                if self.__population.individuals[0].score > best_quality_in_pop:
                    best_quality_in_pop = copy.deepcopy(self.__population.individuals[0].score)
                    best_ever_kernel = copy.deepcopy(self.__population.individuals[0].psf)
                    best_kernels.append(copy.deepcopy(self.__population.individuals[0].psf))
                    upscale_flag = 0

                # селекция и скрещивание
                elite_individuals, non_elite_individuals = self.__population.get_elite_non_elite(self.__elite_count)
                selected_individuals = SelectionOperators.select_tournament(non_elite_individuals, k=len(non_elite_individuals))
                crossed_individuals = CrossoverOperators.uniform_crossover(selected_individuals, probability=self.__crossover_prob)

                # мутация
                mutated_individuals = copy.deepcopy(crossed_individuals[:])
                if True:
                    for ind in mutated_individuals:
                        ind.mutate_smart(self.__mutation_prob, self.__pos_prob)
                else:
                    for ind in mutated_individuals:
                        ind.mutate(self.__mutation_prob, self.__pos_prob)

                # обновление особей популяции
                self.__population.individuals.clear()
                self.__population.individuals.extend(copy.deepcopy(mutated_individuals))
                self.__population.individuals.extend(copy.deepcopy(elite_individuals))
                upscale_flag += 1

            # апскейлим
            if i != len(self.__population.scale_pyramid.psf_sizes) - 1:
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

                best_quality_in_pop = -10000.0
                self.__population.upscale("pad")

            if i == len(self.__population.scale_pyramid.psf_sizes) - 1:
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

        return ImgDeconv.do_deconv(self.__population.blurred, self.__population.individuals[0].psf, type=self.__deconv_type)
