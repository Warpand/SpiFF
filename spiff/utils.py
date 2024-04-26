from typing import Iterable, Iterator, Tuple, TypeVar

T = TypeVar("T")


def triple_wise(iterable: Iterable[T]) -> Iterator[Tuple[T, T, T]]:
    it = iter(iterable)
    while True:
        try:
            one = next(it)
            two = next(it)
            three = next(it)
        except StopIteration:
            break
        yield one, two, three
