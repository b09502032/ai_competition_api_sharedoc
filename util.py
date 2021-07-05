import builtins
import pathlib
import random

import numpy as np
import torch


def same_seeds(seed, use_deterministic_algorithms=False):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if use_deterministic_algorithms is True:
        torch.use_deterministic_algorithms(True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class Logger:
    def __init__(self, *files) -> None:
        self.files = files

    def print(self, *values, sep=' ', end='\n', flush=False) -> None:
        for file in self.files:
            if isinstance(file, (str, pathlib.Path)):
                with open(file, 'a') as f:
                    builtins.print(*values, sep=sep, end=end, file=f, flush=True)
            else:
                builtins.print(*values, sep=sep, end=end, file=file, flush=flush)
