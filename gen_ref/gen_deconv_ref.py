from gen.crossover import *
from gen.mutation import *
from gen.selection import *
from gen_ref.ref_population import RefPopulation
from utils.deconv import do_wiener_deconv_1c
from utils.misc import *
from utils.drawing import *
from utils.size_utils import *

# константы
STAGNATION_POPULATION_COUNT = 30
UPSCALE_TYPE = "pad"
NO_REF_METRIC = "fourier"
REF_METRIC = "ssim"
DECONV_TYPE = "LR"
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.1
POS_PROBABILITY = 0.5
SMART_MUTATION = True


def gen_deblur_image(sharp, blurred, kernel_size=23, elite_count=1, display_process=False):
    population = RefPopulation(sharp, blurred, kernel_size, NO_REF_METRIC, REF_METRIC, DECONV_TYPE)
    best_quality_in_pop = -10000.0
    upscale_flag = 0
    best_ever_kernel = np.zeros(sharp.shape)
    best_kernels = []

    for i in range(0, len(population.kernel_sizes), 1):
        while True:
            if upscale_flag == STAGNATION_POPULATION_COUNT:
                upscale_flag = 0
                break

            population.fit()

            if display_process:
                cv.imshow("best kernel resized", cv.resize(copy.deepcopy(population.individuals[0].psf), None, fx=10, fy=10, interpolation=cv.INTER_AREA))
                cv.imshow("best ever kernel", cv.resize(best_ever_kernel, None, fx=10, fy=10, interpolation=cv.INTER_AREA))
                cv.imshow("best restored", do_wiener_deconv_1c(population.blurred, population.individuals[0].psf, 0.01))
                cv.imshow("sharp", population.sharp)
                cv.imshow("blurred", population.blurred)
                cv.waitKey(1)

            print(f"best quality in pop: {population.individuals[0].score}, best quality ever: {best_quality_in_pop}")

            if population.individuals[0].score > best_quality_in_pop:
                best_quality_in_pop = copy.deepcopy(population.individuals[0].score)
                best_ever_kernel = copy.deepcopy(population.individuals[0].psf)
                best_kernels.append(copy.deepcopy(population.individuals[0].psf))
                upscale_flag = 0

            elite_individuals = copy.deepcopy(population.individuals[:elite_count])
            non_elite_individuals = copy.deepcopy(population.individuals[elite_count:])
            selected_individuals = select_tournament(non_elite_individuals, k=len(non_elite_individuals))
            crossed_individuals = uniform_crossover(selected_individuals, probability=CROSSOVER_PROBABILITY)
            new_individuals = mutate(crossed_individuals, probability=MUTATION_PROBABILITY, add_prob=POS_PROBABILITY, use_smart=SMART_MUTATION)
            population.individuals.clear()
            population.individuals.extend(copy.deepcopy(new_individuals))
            population.individuals.extend(copy.deepcopy(elite_individuals))
            upscale_flag += 1

        # апскейлим
        if i != len(population.kernel_sizes) - 1:
            # xd_test = np.zeros((population.kernel_size, population.kernel_size), np.float32)
            # for kernel in best_kernels:
            #     xd_test += kernel
            # xd_test += population.individuals[0].psf
            # cv.normalize(xd_test, xd_test, 0.0, 1.0, cv.NORM_MINMAX)
            # best_kernels.clear()
            # cv.imshow("xd_test", cv.resize(xd_test, None, fx=10, fy=10, interpolation=cv.INTER_AREA))

            # ЗАПИСЬ
            best_result_for_size = do_wiener_deconv_1c(population.blurred, population.individuals[0].psf, 0.01)
            cv.normalize(best_result_for_size, best_result_for_size, 0.0, 255.0, cv.NORM_MINMAX)

            lel_test = copy.deepcopy(population.individuals[0].psf)
            best_kernel_for_size = cv.resize(lel_test, None, fx=10, fy=10, interpolation=cv.INTER_AREA)
            cv.normalize(best_kernel_for_size, best_kernel_for_size, 0, 255, cv.NORM_MINMAX)
            # cv.imshow("best_kernel_for_size", best_kernel_for_size)

            blurred_normalized = copy.deepcopy(population.blurred)
            cv.normalize(blurred_normalized, blurred_normalized, 0, 255, cv.NORM_MINMAX)


            result_file_name = "../images/results/restored_size_{}.jpg".format(population.kernel_size)
            blurred_file_name = "../images/results/blurred_size_{}.jpg".format(population.kernel_size)
            kernel_file_name = "../images/results/kernel_size_{}.jpg".format(population.kernel_size)

            cv.imwrite(result_file_name, best_result_for_size)
            cv.imwrite(kernel_file_name, best_kernel_for_size)
            cv.imwrite(blurred_file_name, blurred_normalized)
            # ЗАПИСЬ

            # for ind in population.individuals:
            #     ind.psf = copy.deepcopy(xd_test)

            best_quality_in_pop = -10000.0
            population.upscale(UPSCALE_TYPE)

        if i == len(population.kernel_sizes) - 1:
            # xd_test = np.zeros((population.kernel_size, population.kernel_size), np.float32)
            # for kernel in best_kernels:
            #     xd_test += kernel
            # xd_test += population.individuals[0].psf
            # cv.normalize(xd_test, xd_test, 0.0, 1.0, cv.NORM_MINMAX)
            # best_kernels.clear()
            # cv.imshow("xd_test", cv.resize(xd_test, None, fx=10, fy=10, interpolation=cv.INTER_AREA))

            # ЗАПИСЬ
            population.fit()
            best_result_for_size = do_wiener_deconv_1c(population.blurred, population.individuals[0].psf, 0.01)
            cv.normalize(best_result_for_size, best_result_for_size, 0.0, 255.0, cv.NORM_MINMAX)

            lel_test = copy.deepcopy(population.individuals[0].psf)
            best_kernel_for_size = cv.resize(lel_test, None, fx=10, fy=10, interpolation=cv.INTER_AREA)
            cv.normalize(best_kernel_for_size, best_kernel_for_size, 0, 255, cv.NORM_MINMAX)
            # cv.imshow("best_kernel_for_size", best_kernel_for_size)

            blurred_normalized = copy.deepcopy(population.blurred)
            cv.normalize(blurred_normalized, blurred_normalized, 0, 255, cv.NORM_MINMAX)

            result_file_name = "../images/results/restored_size_{}.jpg".format(population.kernel_size)
            blurred_file_name = "../images/results/blurred_size_{}.jpg".format(population.kernel_size)
            kernel_file_name = "../images/results/kernel_size_{}.jpg".format(population.kernel_size)

            cv.imwrite(result_file_name, best_result_for_size)
            cv.imwrite(kernel_file_name, best_kernel_for_size)
            cv.imwrite(blurred_file_name, blurred_normalized)
            # ЗАПИСЬ

    best_psf = population.individuals[0].psf
    return do_wiener_deconv_1c(population.blurred, best_psf, 0.01)


# четкое изображение
sharp = cv.imread("../images/sharp/bstu2.jpg", cv.IMREAD_GRAYSCALE)
#sharp = sharp ** (1/2.2)
sharp = np.float32(sharp)
cv.normalize(sharp, sharp, 0.0, 1.0, cv.NORM_MINMAX)

psf = cv.imread("../images/psfs/2.png", cv.IMREAD_GRAYSCALE)
psf = pad_to_shape(psf, sharp.shape)
blurred = freq_filter_1c(sharp, psf)
cv.normalize(blurred, blurred, 0.0, 1.0, cv.NORM_MINMAX)

# генетика
result = gen_deblur_image(sharp, blurred, display_process=False)

# вывод результата
#cv.imshow("fiunal_result", result)

print("FINISHED!")
cv.waitKey()