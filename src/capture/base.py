from abc import (
    ABC,
    )
from typing import NewType

import cv2
import numpy as np

from src.capture.model import FrameSize
from src.utils import const


class CaptureSource(ABC):
    _cap: cv2.VideoCapture

    def __init__(self):
        if not self._is_opened():
            raise RuntimeError()

        self._set_src_frame_size()
        self._set_dst_frame_size()

    def _is_opened(self):
        return self._cap.isOpened()

    def _set_src_frame_size(self):
        frame = self._read()

        src_height, src_width, _ = frame.shape
        self._src_frame_size = FrameSize(width=src_width, height=src_height)

    def _set_dst_frame_size(self):
        self._dst_frame_size = FrameSize(
                width=const.PREF_VIDEO_W,
                height=(const.PREF_VIDEO_W * self._src_frame_size.height) // self._src_frame_size.width,
                )

    def _read(self) -> np.ndarray:
        ret, frame = self._cap.read()

        if not ret:
            raise RuntimeError()

        return frame

    def read(self) -> np.ndarray:
        return cv2.resize(
                self._read(),
                (self._dst_frame_size.width, self._dst_frame_size.height),
                fx=0,
                fy=0,
                interpolation=cv2.INTER_NEAREST,
                )

    def skip_frames(self, count: int):
        pass

    def release(self) -> None:
        self._cap.release()

    def fps(self) -> int:
        return self._cap.get(cv2.CAP_PROP_FPS)

    def frame_size(self) -> FrameSize:
        return self._dst_frame_size


CaptureSourceImpl = NewType('CaptureSource', CaptureSource)
