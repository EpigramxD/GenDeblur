from gen_ref.ref_population import PopulationRef
from utils.imgDeconv import ImgDeconv
from utils.size_utils import *
from gen.selectionOperators import SelectionOperators
from gen.crossoverOperators import CrossoverOperators

# константы
STAGNATION_POPULATION_COUNT = 15
UPSCALE_TYPE = "pad"
NO_REF_METRIC = "fourier"
REF_METRIC = "ssim"
DECONV_TYPE = "wiener"
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.1
POS_PROBABILITY = 0.5
SMART_MUTATION = True


def gen_deblur_image(sharp, blurred, min_kernel_size=3, step=2, max_kernel_size=23, elite_count=1, display_process=False):
    population = PopulationRef(sharp, blurred, min_kernel_size, step, max_kernel_size)
    best_quality_in_pop = -10000.0
    upscale_flag = 0
    best_ever_kernel = np.zeros(sharp.shape)
    best_kernels = []

    for i in range(0, len(population.scale_pyramid.kernel_sizes), 1):
        while True:
            if upscale_flag == STAGNATION_POPULATION_COUNT:
                upscale_flag = 0
                break

            population.fit(NO_REF_METRIC, REF_METRIC, DECONV_TYPE)

            if display_process:
                cv.imshow("best kernel resized", cv.resize(copy.deepcopy(population.individuals[0].psf), None, fx=10, fy=10, interpolation=cv.INTER_AREA))
                cv.imshow("best ever kernel", cv.resize(best_ever_kernel, None, fx=10, fy=10, interpolation=cv.INTER_AREA))
                cv.imshow("best restored", ImgDeconv.do_deconv(population.blurred, population.individuals[0].psf, type="wiener"))
                cv.imshow("sharp", population.sharp)
                cv.imshow("blurred", population.blurred)
                cv.waitKey(1)

            print(f"best quality in pop: {population.individuals[0].score}, best quality ever: {best_quality_in_pop}")

            if population.individuals[0].score > best_quality_in_pop:
                best_quality_in_pop = copy.deepcopy(population.individuals[0].score)
                best_ever_kernel = copy.deepcopy(population.individuals[0].psf)
                best_kernels.append(copy.deepcopy(population.individuals[0].psf))
                upscale_flag = 0

            # селекция и скрещивание
            elite_individuals, non_elite_individuals = population.get_elite_non_elite(elite_count)
            selected_individuals = SelectionOperators.select_tournament(non_elite_individuals, k=len(non_elite_individuals))
            crossed_individuals = CrossoverOperators.uniform_crossover(selected_individuals, probability=CROSSOVER_PROBABILITY)

            # мутация
            mutated_individuals = copy.deepcopy(crossed_individuals[:])
            if SMART_MUTATION:
                for ind in mutated_individuals:
                    ind.mutate_smart(MUTATION_PROBABILITY, POS_PROBABILITY)
            else:
                for ind in mutated_individuals:
                    ind.mutate(MUTATION_PROBABILITY, POS_PROBABILITY)

            # обновление особей популяции
            population.individuals.clear()
            population.individuals.extend(copy.deepcopy(mutated_individuals))
            population.individuals.extend(copy.deepcopy(elite_individuals))
            upscale_flag += 1

        # апскейлим
        if i != len(population.scale_pyramid.kernel_sizes) - 1:
            # xd_test = np.zeros((population.kernel_size, population.kernel_size), np.float32)
            # for kernel in best_kernels:
            #     xd_test += kernel
            # xd_test += population.individuals[0].psf
            # cv.normalize(xd_test, xd_test, 0.0, 1.0, cv.NORM_MINMAX)
            # best_kernels.clear()
            # cv.imshow("xd_test", cv.resize(xd_test, None, fx=10, fy=10, interpolation=cv.INTER_AREA))

            # ЗАПИСЬ
            best_result_for_size = ImgDeconv.do_deconv(population.blurred, population.individuals[0].psf, type="wiener")
            cv.normalize(best_result_for_size, best_result_for_size, 0.0, 255.0, cv.NORM_MINMAX)

            lel_test = copy.deepcopy(population.individuals[0].psf)
            best_kernel_for_size = cv.resize(lel_test, None, fx=10, fy=10, interpolation=cv.INTER_AREA)
            cv.normalize(best_kernel_for_size, best_kernel_for_size, 0, 255, cv.NORM_MINMAX)
            # cv.imshow("best_kernel_for_size", best_kernel_for_size)

            blurred_normalized = copy.deepcopy(population.blurred)
            cv.normalize(blurred_normalized, blurred_normalized, 0, 255, cv.NORM_MINMAX)


            result_file_name = "../images/results/restored_size_{}.jpg".format(population.current_psf_size)
            blurred_file_name = "../images/results/blurred_size_{}.jpg".format(population.current_psf_size)
            kernel_file_name = "../images/results/kernel_size_{}.jpg".format(population.current_psf_size)

            cv.imwrite(result_file_name, best_result_for_size)
            cv.imwrite(kernel_file_name, best_kernel_for_size)
            cv.imwrite(blurred_file_name, blurred_normalized)
            # ЗАПИСЬ

            # for ind in population.individuals:
            #     ind.psf = copy.deepcopy(xd_test)

            best_quality_in_pop = -10000.0
            population.upscale(UPSCALE_TYPE)

        if i == len(population.scale_pyramid.kernel_sizes) - 1:
            # xd_test = np.zeros((population.kernel_size, population.kernel_size), np.float32)
            # for kernel in best_kernels:
            #     xd_test += kernel
            # xd_test += population.individuals[0].psf
            # cv.normalize(xd_test, xd_test, 0.0, 1.0, cv.NORM_MINMAX)
            # best_kernels.clear()
            # cv.imshow("xd_test", cv.resize(xd_test, None, fx=10, fy=10, interpolation=cv.INTER_AREA))

            # ЗАПИСЬ
            population.fit(NO_REF_METRIC, REF_METRIC, DECONV_TYPE)
            best_result_for_size = ImgDeconv.do_deconv(population.blurred, population.individuals[0].psf, type="wiener")
            cv.normalize(best_result_for_size, best_result_for_size, 0.0, 255.0, cv.NORM_MINMAX)

            lel_test = copy.deepcopy(population.individuals[0].psf)
            best_kernel_for_size = cv.resize(lel_test, None, fx=10, fy=10, interpolation=cv.INTER_AREA)
            cv.normalize(best_kernel_for_size, best_kernel_for_size, 0, 255, cv.NORM_MINMAX)
            # cv.imshow("best_kernel_for_size", best_kernel_for_size)

            blurred_normalized = copy.deepcopy(population.blurred)
            cv.normalize(blurred_normalized, blurred_normalized, 0, 255, cv.NORM_MINMAX)

            result_file_name = "../images/results/restored_size_{}.jpg".format(population.current_psf_size)
            blurred_file_name = "../images/results/blurred_size_{}.jpg".format(population.current_psf_size)
            kernel_file_name = "../images/results/kernel_size_{}.jpg".format(population.current_psf_size)

            cv.imwrite(result_file_name, best_result_for_size)
            cv.imwrite(kernel_file_name, best_kernel_for_size)
            cv.imwrite(blurred_file_name, blurred_normalized)
            # ЗАПИСЬ

    return ImgDeconv.do_deconv(population.blurred, population.individuals[0].psf, type="wiener")


# четкое изображение
sharp = cv.imread("../images/sharp/bstu2.jpg", cv.IMREAD_GRAYSCALE)
#sharp = sharp ** (1/2.2)
sharp = np.float32(sharp)
cv.normalize(sharp, sharp, 0.0, 1.0, cv.NORM_MINMAX)

psf = cv.imread("../images/psfs/2.png", cv.IMREAD_GRAYSCALE)
psf = ImgUtils.pad_to_shape(psf, sharp.shape)
blurred = ImgUtils.freq_filter(sharp, psf)
cv.normalize(blurred, blurred, 0.0, 1.0, cv.NORM_MINMAX)

# генетика
result = gen_deblur_image(sharp, blurred, display_process=False)

# вывод результата
#cv.imshow("fiunal_result", result)

print("FINISHED!")
cv.waitKey()