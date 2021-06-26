import random
import copy
"""
Методы селекции
"""


def select_random(individuals, k):
    """
    Выбрать k случайных особей
    :param individuals: список особей
    :param k: количество особей для выбора
    :returns: выбранные особи в виде списка
    """
    return [random.choice(individuals) for i in range(k)]


def select_tournament(individuals, k, tournsize=2):
    """
    Турнирная селекция
    :param individuals: популяция
    :param k: количество особей для выбора
    :param tournsize: размер турнира
    :return: отобранные особи
    """
    chosen = []
    for i in range(k):
        aspirants = select_random(individuals, tournsize)
        chosen.append(copy.deepcopy(max(aspirants, key=lambda ind: ind.score)))
    return chosen