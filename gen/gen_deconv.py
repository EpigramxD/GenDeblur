import random

import cv2 as cv
from gen.individual import Individual
from gen.population import Population
from gen.selection import *
from gen.crossover import *
from gen.mutation import *
from gen.crossover import crossover_two_individuals
from utils.drawing import *
from gen.size_utils import *
from utils.deconv import do_RL_deconv, do_weiner_deconv
from deap import tools

# коилчество популяций, при которых подряд не будет происходить улучшение качества
# если будет достигнуто это количество, то произойдет увеличение ядра популяции + апскейл изображений
STAGNATION_POPULATION_COUNT = 200
UPSCALE_TYPE = "pad"
DECONV_TYPE = "weiner"


def gen_deblur_image(image, kernel_size=23, metric_type="brisque", elite_count=3):
    # создание новой популяции с максимальным размером ядра 23
    population = Population(image, kernel_size, metric_type)
    # лучшее найденное качество (лучший показатель целевой функции)
    best_quality_in_pop = -10000.0
    # флаг, который отвечает за увеличение ядра популяции (сравнивается с STAGNATION_POPULATION_COUNT)
    upscale_flag = 0

    # для всех размеров ядер
    for i in range(0, len(population.kernel_sizes), 1):
        while True:
            if upscale_flag == STAGNATION_POPULATION_COUNT:
                upscale_flag = 0
                break
            # оценка приспособленности
            population.fit(DECONV_TYPE)
            # вывод наилучшей особи
            cv.imshow("best kernel resized", cv.resize(copy.deepcopy(population.individuals[0].psf), None, fx=10, fy=10, interpolation=cv.INTER_AREA))
            #cv.imshow("best restored", do_RL_deconv(population.image, population.individuals[0].psf, iterations=10))
            cv.imshow("best restored", do_weiner_deconv(population.image, population.individuals[0].psf, 100))
            print(f"best quality in pop: {population.individuals[0].score}, best quality ever: {best_quality_in_pop}")
            cv.waitKey(10)

            if population.individuals[0].score > best_quality_in_pop:
                best_quality_in_pop = population.individuals[0].score
                upscale_flag = 0

            # элитизм
            elite_individuals = copy.deepcopy(population.individuals[:elite_count])
            non_elite_individuals = copy.deepcopy(population.individuals[elite_count:])
            selected_individuals = select_tournament(non_elite_individuals, len(non_elite_individuals))
            crossed_individuals = uniform_crossover(selected_individuals, probability=0.9)
            new_individuals = mutate(crossed_individuals, probability=0.08)
            population.individuals.clear()
            population.individuals.extend(copy.deepcopy(new_individuals))
            population.individuals.extend(copy.deepcopy(elite_individuals))

            # увеличиваем флаг
            #upscale_flag += 1

        # апскейлим
        if i != len(population.kernel_sizes) - 1:
            population.fit(DECONV_TYPE)
            population.upscale(UPSCALE_TYPE)


image = cv.imread("D:\Images\IMGS\pic6.jpg", cv.IMREAD_GRAYSCALE)
image = np.float32(image)
cv.normalize(image, image, 0.0, 1.0, cv.NORM_MINMAX)

gen_deblur_image(image)

cv.waitKey()