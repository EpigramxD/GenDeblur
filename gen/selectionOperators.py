import random
import copy

class SelectionOperators(object):
    @staticmethod
    def __select_random(individuals, k):
        """
        Выбрать k случайных особей
        :param individuals: особои
        :param k: количество особей для выбора
        :returns: выбранные особи в виде списка
        """
        return [random.choice(individuals) for i in range(k)]

    @staticmethod
    def select_tournament(individuals, k, tournsize=2):
        """
        Турнирная селекция
        :param individuals: особи
        :param k: количество особей для выбора
        :param tournsize: размер турнира
        :return: отобранные особи
        """
        chosen = []
        for i in range(k):
            aspirants = SelectionOperators.__select_random(individuals, tournsize)
            chosen.append(copy.deepcopy(max(aspirants, key=lambda ind: ind.score)))
        return chosen