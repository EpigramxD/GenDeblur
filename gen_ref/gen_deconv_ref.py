from gen.crossover import *
from gen.mutation import *
from gen_ref.ref_population import RefPopulation
from gen.selection import *
from utils.size_utils import *
from utils.deconv import do_weiner_deconv_1c

# константы
STAGNATION_POPULATION_COUNT = 500
UPSCALE_TYPE = "pod"
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.0005
DECONV_TYPE = "weiner"


def gen_deblur_image(sharp, blurred, kernel_size=23, elite_count=0):
    population = RefPopulation(sharp, blurred, kernel_size)
    best_quality_in_pop = -10000.0
    upscale_flag = 0
    best_ever_kernel = np.zeros(sharp.shape)

    for i in range(0, len(population.kernel_sizes), 1):
        while True:
            if upscale_flag == STAGNATION_POPULATION_COUNT:
                upscale_flag = 0
                break
            population.fit(DECONV_TYPE)
            cv.imshow("best kernel resized", cv.resize(copy.deepcopy(population.individuals[0].psf), None, fx=10, fy=10, interpolation=cv.INTER_AREA))
            cv.imshow("best ever kernel", cv.resize(best_ever_kernel, None, fx=10, fy=10, interpolation=cv.INTER_AREA))
            #cv.imshow("best restored", do_RL_deconv(population.image, population.individuals[0].psf, iterations=10))
            cv.imshow("best restored", do_weiner_deconv_1c(population.blurred, population.individuals[0].psf, 100))
            cv.imshow("sharp", population.sharp)
            cv.imshow("blurred", population.blurred)
            print(f"best quality in pop: {population.individuals[0].score}, best quality ever: {best_quality_in_pop}")
            cv.waitKey(10)

            if population.individuals[0].score > best_quality_in_pop:
                best_quality_in_pop = population.individuals[0].score
                best_ever_kernel = population.individuals[0].psf
                upscale_flag = 0

            # элитизм
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
            best_quality_in_pop = -10000.0
            population.fit(DECONV_TYPE)
            population.individuals[0].psf = best_ever_kernel
            population.upscale(UPSCALE_TYPE)
            population.fit(DECONV_TYPE)


sharp = cv.imread("../images/sharp/1.jpg", cv.IMREAD_GRAYSCALE)
blurred = cv.imread("../images/blurred/1.jpg", cv.IMREAD_GRAYSCALE)

sharp = np.float32(sharp)
cv.normalize(sharp, sharp, 0.0, 1.0, cv.NORM_MINMAX)

blurred = np.float32(blurred)
cv.normalize(blurred, blurred, 0.0, 1.0, cv.NORM_MINMAX)

gen_deblur_image(sharp, blurred)
cv.waitKey()