from cv2 import cv2

from .base import CaptureSource
from ..utils import const


class Camera(CaptureSource):
    def __init__(self, device_number: int):
        self._cap = cv2.VideoCapture(device_number, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FPS, const.PREF_FPS)

        super(Camera, self).__init__()
