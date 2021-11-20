import numpy as np
import copy
import random


class MutationOperators(object):

    @staticmethod
    def smartMutation(individuals, probability, pos_prob):
        """
        Умная мутация особи
        :param individuals - мутируемые особи
        :param probability: вероятность мутирования
        :param pos_prob: вероятность добавления рандомного значения к гену, если меньше, то вычитание
        """
        mutated_individuals = copy.deepcopy(individuals)
        for individual in mutated_individuals:
            bright_pixels = np.argwhere(individual.psf > 0.1)
            for position in bright_pixels:
                if random.random() < probability:
                    random_neighbor_position = individual.get_random_neighbor_position(position[0], position[1])

                    if random.random() < pos_prob:
                        try:
                            individual.psf[random_neighbor_position[0], random_neighbor_position[1]] += random.uniform(0.1,
                                                                                                                   1.0)
                            if individual.psf[random_neighbor_position[0], random_neighbor_position[1]] > 1.0:
                                individual.psf[random_neighbor_position[0], random_neighbor_position[1]] = 1.0
                        except IndexError:
                            individual.psf[position[0], position[1]] += random.uniform(0.1, 1.0)
                            if individual.psf[position[0], position[1]] > 1.0:
                                individual.psf[position[0], position[1]] = 1.0
                    else:
                        individual.psf[position[0], position[1]] -= random.uniform(0.1, 1.0)
                        if individual.psf[position[0], position[1]] < 0:
                            individual.psf[position[0], position[1]] = 0
                    break
            individual.normalize()

        return mutated_individuals

    @staticmethod
    def mutate(individuals, type="smart"):
        pass
