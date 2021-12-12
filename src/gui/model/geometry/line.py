from dataclasses import dataclass

from .point import Point


@dataclass
class Line:
    a: Point
    b: Point
