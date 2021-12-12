import colorsys
from typing import (
    NewType,
    Optional,
    )

import numpy as np

from src.gui.model import Color
from src.gui.model.geometry import Point
from src.utils import const

_hsv_colors = [(i / len(const.CLASS_NAMES_USED), 1, 1.0) for i in range(len(const.CLASS_NAMES_USED))]
_gbr_colors = []
for h, s, v in _hsv_colors:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    _gbr_colors.append(Color((g * 255, b * 255, r * 255)))


class Vehicle:
    _type_id: int
    _caption: str

    def __init__(self):
        self._color = _gbr_colors[self._type_id]

        self._centroid: Point = Point(0, 0)
        self._score: float = 0.0
        self._mask: np.ndarray = None

    def caption(self) -> str:
        return self._caption

    def color(self) -> Color:
        return self._color

    def set_color(self, color: Color):
        self._color = color

    def centroid(self) -> Point:
        return self._centroid

    def set_centroid(self, point: Point):
        self._centroid = point

    def score(self) -> float:
        return self._score

    def set_score(self, score: float):
        self._score = score

    def mask(self) -> np.ndarray:
        return self._mask

    def set_mask(self, mask: np.ndarray):
        self._mask = mask


VehicleImpl = NewType('VehicleImpl', Vehicle)


class Bicycle(Vehicle):
    _type_id = 0
    _caption = 'Bicycle'


class Car(Vehicle):
    _type_id = 1
    _caption = 'Car'


class Motorcycle(Vehicle):
    _type_id = 2
    _caption = 'Motorcycle'


class Truck(Vehicle):
    _type_id = 3
    _caption = 'Truck'


class Bus(Vehicle):
    _type_id = 4
    _caption = 'Bus'


def vehicle_from_class_id(class_id: int) -> Optional[VehicleImpl]:
    class_name = const.CLASS_NAMES_ALL[class_id]

    if class_name == 'bicycle':
        return Bicycle()
    elif class_name == 'car':
        return Car()
    elif class_name == 'motorcycle':
        return Motorcycle()
    elif class_name == 'truck':
        return Truck()
    elif class_name == 'bus':
        return Bus()

    return None
