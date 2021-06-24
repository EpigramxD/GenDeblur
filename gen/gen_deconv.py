from gen.crossover import *
from gen.mutation import *
from gen.population import Population
from gen.selection import *
from utils.size_utils import *
from utils.deconv import do_weiner_deconv_1c, do_RL_deconv

# константы
STAGNATION_POPULATION_COUNT = 200
UPSCALE_TYPE = "pad"
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.05
DECONV_TYPE = "weiner"


def gen_deblur_image(image, kernel_size=23, metric_type="fourier", elite_count=0):
    population = Population(image, kernel_size, metric_type)
    best_quality_in_pop = -10000.0
    upscale_flag = 0

    for i in range(0, len(population.kernel_sizes), 1):
        while True:
            if upscale_flag == STAGNATION_POPULATION_COUNT:
                upscale_flag = 0
                break
            population.fit(DECONV_TYPE)
            cv.imshow("best kernel resized", cv.resize(copy.deepcopy(population.individuals[0].psf), None, fx=10, fy=10, interpolation=cv.INTER_AREA))
            cv.imshow("best restored", do_weiner_deconv_1c(population.image, population.individuals[0].psf, 100))
            print(f"best quality in pop: {population.individuals[0].score}, best quality ever: {best_quality_in_pop}")
            cv.waitKey(10)

            if population.individuals[0].score > best_quality_in_pop:
                best_quality_in_pop = population.individuals[0].score
                upscale_flag = 0

            elite_individuals = copy.deepcopy(population.individuals[:elite_count])
            non_elite_individuals = copy.deepcopy(population.individuals[elite_count:])
            selected_individuals = select_tournament(non_elite_individuals, len(non_elite_individuals))
            crossed_individuals = uniform_crossover(selected_individuals, probability=CROSSOVER_PROBABILITY)
            new_individuals = mutate(crossed_individuals, probability=MUTATION_PROBABILITY)
            population.individuals.clear()
            population.individuals.extend(copy.deepcopy(new_individuals))
            population.individuals.extend(copy.deepcopy(elite_individuals))
            upscale_flag += 1

        # апскейлим
        if i != len(population.kernel_sizes) - 1:
            population.fit(DECONV_TYPE)
            population.upscale(UPSCALE_TYPE)


image = cv.imread("D:\Images\IMGS\pic6.jpg", cv.IMREAD_GRAYSCALE)
image = np.float32(image)
cv.normalize(image, image, 0.0, 1.0, cv.NORM_MINMAX)

gen_deblur_image(image)

cv.waitKey()