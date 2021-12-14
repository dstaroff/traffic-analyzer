import numpy as np
from cv2 import cv2

from .base import CaptureSource


class Video(CaptureSource):
    def __init__(self, file_path: str):
        self._cap = cv2.VideoCapture(file_path)

        super(Video, self).__init__()

    def _read(self) -> np.ndarray:
        if self._cap.get(cv2.CAP_PROP_POS_FRAMES) >= self._cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return super(Video, self)._read()

    def skip_frames(self, count: int):
        if count == 0:
            return

        offset_frame = (self._cap.get(cv2.CAP_PROP_POS_FRAMES) + count) % self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, offset_frame)
