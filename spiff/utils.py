import io
from typing import Iterable, Iterator, Tuple, TypeVar

import matplotlib.figure
import wandb
from PIL import Image

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


def figure_to_wandb(fig: matplotlib.figure.Figure) -> wandb.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, bbox_inches="tight")

    with Image.open(buffer) as img:
        return wandb.Image(img)
