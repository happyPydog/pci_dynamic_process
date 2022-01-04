r"""Decorator interface"""
from typing import Callable, Any
import time
from functools import wraps
import numpy as np


class Decimals:
    """Round decorator."""

    def __init__(self, *, decimals: int) -> None:
        self.__decimals = decimals

    def __call__(self, func) -> Callable[..., np.ndarray]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> np.ndarray:
            return np.round(func(*args, **kwargs), self.__decimals)

        return wrapper


class RunTime:
    """Run time decorator."""

    def __call__(self, func) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> None:
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} Run times: {end - start}/s")

        return wrapper
