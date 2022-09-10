import enum
import math
import typing


@enum.unique
class DominationStatus(enum.Enum):
    LEFT_DOMINATES_RIGHT = enum.auto()
    RIGHT_DOMINATES_LEFT = enum.auto()
    NO_DOMINATION = enum.auto()


Index = typing.NewType("Index", int)
DominationTable = dict[tuple[Index, Index], DominationStatus]
Fitness = tuple[float, ...]
ParetoFront = list[Index]


def calc_domination_status(lhs: Fitness, rhs: Fitness) -> DominationStatus:
    assert len(lhs) == len(rhs)

    any_less = any(left < right for left, right in zip(lhs, rhs))
    any_greater = any(left > right for left, right in zip(lhs, rhs))

    match (any_less, any_greater):
        case False, False:
            return DominationStatus.NO_DOMINATION
        case False, True:
            return DominationStatus.LEFT_DOMINATES_RIGHT
        case True, False:
            return DominationStatus.RIGHT_DOMINATES_LEFT
        case True, True:
            return DominationStatus.NO_DOMINATION
        case _:
            raise ValueError("should never be reached")


def validate_fitnesses(fitnesses: typing.Sequence[Fitness]) -> None:
    assert len(fitnesses) >= 1

    num_objectives = len(fitnesses[0])
    assert num_objectives >= 1
    assert all(len(f) == num_objectives for f in fitnesses)


def calc_domination_table(fitnesses: typing.Sequence[Fitness]) -> DominationTable:
    validate_fitnesses(fitnesses)

    table = {}

    for index_left, fitness_left in enumerate(fitnesses):
        for index_right, fitness_right in enumerate(fitnesses):
            status = calc_domination_status(fitness_left, fitness_right)
            table_key = (Index(index_left), Index(index_right))
            table[table_key] = status

    return table


def is_non_dominated(
    target: Index,
    competitors: list[Index],
    domination_table: DominationTable,
) -> bool:
    domination_statuses = (domination_table[(target, ci)] for ci in competitors)
    return not any(
        ds == DominationStatus.RIGHT_DOMINATES_LEFT for ds in domination_statuses
    )


def find_non_dominated(
    competitors: list[Index],
    domination_table: DominationTable,
) -> ParetoFront:
    assert len(competitors) >= 1

    front = [
        index
        for index in competitors
        if is_non_dominated(index, competitors, domination_table)
    ]

    return ParetoFront(front)


def calc_pareto_fronts(fitnesses: typing.Sequence[Fitness]) -> list[ParetoFront]:
    """
    Returns a list of `ParetoFront`s, each one being a list of
    indices of elements of `fitnesses`.

    This function assumes that objectives must be maximized.
    """

    validate_fitnesses(fitnesses)

    domination_table = calc_domination_table(fitnesses)

    fronts = []

    competitors = [Index(idx) for idx in range(len(fitnesses))]
    while competitors:
        front = find_non_dominated(competitors, domination_table)
        fronts.append(front)
        competitors = [i for i in competitors if i not in front]

    return fronts
