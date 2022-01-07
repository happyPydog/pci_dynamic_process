from dataclasses import dataclass


@dataclass
class Parameters:
    mean: float
    sigma: float
    USL: float
    LSL: float
