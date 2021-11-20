import random
import numpy as np

class CrossoverOperators(object):
    @staticmethod
    def __uniform_crossover(selected_individuals, probability):
        """
        Равномерное скрещивание особей
        :param selected_individuals: отобранные на этапе селекции особи
        :param probability: вероятность скрещивания
        :return: результат скрещивания (список особей)
        """
        if len(selected_individuals) != 0:
            psf_size = selected_individuals[0].psf.shape[0]
            selected = selected_individuals[:]

            for parent1, parent2 in zip(selected[::2], selected[1::2]):
                psf1 = parent1.psf.flatten()
                psf2 = parent2.psf.flatten()

                size = len(psf1)
                for i in range(size):
                    if random.random() < probability:
                        psf1[i], psf2[i] = psf2[i], psf1[i]

                parent1.psf = np.reshape(psf1, (psf_size, psf_size))
                parent2.psf = np.reshape(psf2, (psf_size, psf_size))

            return selected

    @staticmethod
    def crossover(selected_individuals, crossover_args):
        try:
            type = crossover_args["type"]
            probability = crossover_args["probability"]
            if not isinstance(type, str) and not isinstance(probability, float):
                raise AttributeError("Crossover type should be string and prob should be float")
        except KeyError:
            raise AttributeError("Define crossover type and probability")

        if type == "uniform":
            return CrossoverOperators.__uniform_crossover(selected_individuals, probability)