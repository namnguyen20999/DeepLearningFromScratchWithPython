from typing import Callable
from numpy import ndarray

type ArrayFunction = Callable[[ndarray], ndarray]
type Chain = list[ArrayFunction]
