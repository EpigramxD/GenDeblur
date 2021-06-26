import copy


def mutate(crossed_individuals, probability=0.1, add_prob=0.5, use_smart=False):
    """
    Мутация скрещенных особей
    Используется метод mutate класса Individual
    :param crossed_population: скрещенные особи
    :param probability: вероятность скрещивания
    :return: мутировання популяция
    """
    mutated_individuals = copy.deepcopy(crossed_individuals[:])
    if use_smart:
        for ind in mutated_individuals:
            ind.mutate_smart(probability, add_prob)
    else:
        for ind in mutated_individuals:
            ind.mutate(probability, add_prob)
    return mutated_individuals
