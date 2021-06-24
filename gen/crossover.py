import random
import numpy as np
import cv2 as cv
from gen.individual import Individual

"""
Методы скрещивания
"""


def crossover_two_individuals(ind1, ind2):
    """
    TODO: убрать, это нерабочий бред
    Экспериментальное скрещивание, которое пробовал в самом начале
    Используется в функции crossover
    :param ind1: родитель 1
    :param ind2: родитель 2
    :return: две новые особи
    """
    child1 = Individual(ind1.kernel_size, False)
    child2 = Individual(ind1.kernel_size, False)
    child1.psf = ind1.psf * ind2.psf
    child2.psf = ind1.psf * ind2.psf
    child1.normalize()
    child2.normalize()
    return child1, child2


def crossover(selected_individuals, size, probability=0.5):
    """
    TODO: убрать, это нерабочий бред
    Скрещивание отобранных особей
    :param selected_individuals: отобранные на этапе селекции особи
    :param size: количество особей, которые должны получиться в результате скрещивания
    :param probability: вероятность скрещивания
    :return: результат скрещивания (список особей)
    """
    selected = selected_individuals.copy()
    new_population = []

    for parent1, parent2 in zip(selected[::2], selected[1::2]):
        if len(new_population) == size:
            break

        if random.random() < probability:
            children = crossover_two_individuals(parent1, parent2)
            new_population.append(children[0])
            new_population.append(children[1])
            #selected.remove(parent1)
            #selected.remove(parent2)

    if len(new_population) < size:
        need_to_fill = size - len(new_population)
        new_population.extend(selected[:need_to_fill])

    return new_population


def uniform_crossover(selected_individuals, probability):
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