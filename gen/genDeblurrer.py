import copy

import cv2 as cv
import numpy as np

from gen_ref.ref_population import PopulationRef
from utils.imgDeconv import ImgDeconv
from utils.imgUtils import ImgUtils
from utils.scalePyramid import ScalePyramidRef
from .crossoverOperators import CrossoverOperators
from .selectionOperators import SelectionOperators
from .mutationOperators import MutationOperators


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
                 elite_count):
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

    def deblur(self, sharp_img, blurred_img):
        sharp_img_gray = ImgUtils.to_grayscale(sharp_img)
        sharp_img_gray = ImgUtils.im2double(sharp_img_gray)

        blurred_img_gray = ImgUtils.to_grayscale(blurred_img)
        blurred_img_gray = ImgUtils.im2double(blurred_img_gray)

        scale_pyramid = ScalePyramidRef(sharp_img_gray, blurred_img_gray, self.__pyramid_args["min_psf_size"], self.__pyramid_args["step"], self.__pyramid_args["max_psf_size"])
        self.__population = PopulationRef(scale_pyramid, expand_factor=self.__population_expand_factor)

        best_quality_in_pop = -10000.0
        upscale_flag = 0
        best_ever_kernel = np.zeros(sharp_img_gray.shape)
        best_kernels = []

        for i in range(0, len(scale_pyramid.psf_sizes), 1):
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

                # селекция
                elite_individuals, non_elite_individuals = self.__population.get_elite_non_elite(self.__elite_count)
                self.__selection_args[0]["k"] = len(non_elite_individuals)
                selected_individuals = SelectionOperators.select(non_elite_individuals, self.__selection_args[0])
                # скрещивание
                crossed_individuals = CrossoverOperators.crossover(selected_individuals, self.__crossover_args[0])
                # мутация
                mutated_individuals = MutationOperators.mutate(crossed_individuals, self.__mutation_args[0])

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

                best_quality_in_pop = -10000.0
                self.__population.upscale("pad")

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

        return ImgDeconv.do_deconv(self.__population.blurred, self.__population.individuals[0].psf, type=self.__deconv_type)
