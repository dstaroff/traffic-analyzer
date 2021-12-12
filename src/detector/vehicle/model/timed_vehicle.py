import datetime
from functools import lru_cache
from typing import (
    List,
    Optional,
    )

from src.gui.model.geometry import Point
from .vehicle import VehicleImpl


class TimedVehicle:
    def __init__(self, vehicle: VehicleImpl):
        self.vehicle = vehicle

        self.field_entered_time: Optional[datetime.datetime] = None
        self.field_exited_time: Optional[datetime.datetime] = None
        self.trace: List[Point] = []

    def has_entered_field(self) -> bool:
        return self.field_entered_time is not None

    def has_exited_field(self) -> bool:
        return self.field_exited_time is not None

    @lru_cache(maxsize=None)
    def speed(self, distance: int) -> float:
        """ Returns calculated vehicle speed based on distance divided by time delta
        :param distance: passed distance in meters
        :return: vehicle speed in km/h
        """
        assert [self.has_entered_field(), self.has_exited_field()]
        time_delta = self.field_exited_time - self.field_entered_time
        meters_per_second = distance / time_delta.total_seconds()
        kilometers_per_hour = meters_per_second * 3.6

        return kilometers_per_hour
