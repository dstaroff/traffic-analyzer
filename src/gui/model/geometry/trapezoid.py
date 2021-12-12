from dataclasses import dataclass

from .line import Line


@dataclass
class Trapezoid:
    major: Line
    minor: Line
