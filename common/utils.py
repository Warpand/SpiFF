from datetime import datetime

import coolname
import torch.nn


def get_run_name(base: str) -> str:
    if base:
        name = base
    else:
        name = coolname.generate_slug(2)
    return f"{name}-{datetime.now().strftime('%d.%m.%Y-%H:%M')}"


def freeze(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
