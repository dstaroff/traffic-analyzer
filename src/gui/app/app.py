import datetime
import math
from typing import (
    List,
    Tuple,
    Type,
    Union,
    )

import numpy as np
import qimage2ndarray
from cv2 import cv2
from PySide2.QtCore import (
    QRunnable,
    Qt,
    QThreadPool,
    Signal,
    )
from PySide2.QtGui import (
    QPixmap,
    )
from PySide2.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
    )

from src.capture import (
    Camera,
    Video,
    )
from src.capture.base import (
    CaptureSource,
    CaptureSourceImpl,
    )
from src.capture.model import FrameSize
from src.detector import VehicleDetector
from src.detector.vehicle.model import (
    TimedVehicle,
    VehicleImpl,
    )
from src.gui.model.geometry import (
    Line,
    Point,
    Trapezoid,
    )
from src.utils import const

radius = 5


# noinspection PyAttributeOutsideInit
class App(QApplication):
    _update_pixmap_signal: Signal = Signal()
    _update_traffic_load_signal: Signal = Signal()

    def __init__(self):
        super(App, self).__init__()
        self.setApplicationName(const.APP_NAME)
        self.setApplicationDisplayName(const.APP_DISPLAY_NAME)
        self._window = QMainWindow()

        self._debug_mode = False
        self._capture: CaptureSource = None
        self._detector = VehicleDetector()

        self._pixmap: QPixmap = None
        self._traffic_load: float = 0.0
        self._field = Trapezoid(Line(Point(0, 0), Point(0, 0)), Line(Point(0, 0), Point(0, 0)))
        self._field_points = np.array(
                [[
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    ]], dtype='int32'
                )
        self._field_mask: np.ndarray = None
        self._last_frame_time: datetime.datetime = None
        self._last_frame_vehicles: List[TimedVehicle] = []

        self._setup_ui()

    def _set_debug_mode(self, debug_mode: bool):
        self._debug_mode = debug_mode
        if self._debug_mode:
            self.setApplicationDisplayName(f'{const.APP_DISPLAY_NAME} [DEBUG MODE]')
        else:
            self.setApplicationDisplayName(const.APP_DISPLAY_NAME)

    def _setup_ui(self):
        self._setup_menu()

        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)

        video_control_container = QWidget()
        video_control_layout = QHBoxLayout(video_control_container)
        video_control_layout.addLayout(self._setup_video_ui())
        video_control_layout.addLayout(self._setup_control_ui())

        info_container = QWidget()
        info_layout = QHBoxLayout(info_container)
        self._traffic_load_text = QLabel('Traffic load: 0%')
        info_layout.addWidget(self._traffic_load_text)

        main_layout.addWidget(video_control_container)
        main_layout.addWidget(info_container)

        self._window.setCentralWidget(main_container)

    def _setup_menu(self):
        menu = self._window.menuBar()
        capture_menu = menu.addMenu('Capture')
        utils_menu = menu.addMenu('Utils')

        camera_capture = QAction('From Camera', self._window)
        camera_capture.setShortcut('Ctrl+C')
        camera_capture.triggered.connect(lambda: self.set_capture(Camera(0)))

        video_capture = QAction('From Video', self._window)
        video_capture.setShortcut('Ctrl+V')
        video_capture.triggered.connect(self._set_video_capture)

        debug = QAction('Debug mode', self._window)
        debug.setShortcut('Ctrl+D')
        debug.triggered.connect(lambda: self._set_debug_mode(not self._debug_mode))

        capture_menu.addActions([camera_capture, video_capture])
        utils_menu.addAction(debug)

    def _setup_video_ui(self) -> QLayout:
        layout = QHBoxLayout()
        self._video = QLabel('No Camera Feed')
        layout.addWidget(self._video)

        return layout

    def _setup_control_ui(self) -> QLayout:
        layout = QVBoxLayout()

        start_line_label = QLabel('Start line coords')
        start_line_fst_label = QLabel('Start point')
        self._start_line_a_x_slider, start_line_fst_x_layout = self._create_slider('X: ')
        self._start_line_a_y_slider, start_line_fst_y_layout = self._create_slider('Y: ')
        start_line_snd_label = QLabel('End point')
        self._start_line_b_x_slider, start_line_snd_x_layout = self._create_slider('X: ')
        self._start_line_b_y_slider, start_line_snd_y_layout = self._create_slider('Y: ')

        layout.addWidget(start_line_label, alignment=Qt.AlignCenter)
        layout.addWidget(start_line_fst_label)
        layout.addLayout(start_line_fst_x_layout)
        layout.addLayout(start_line_fst_y_layout)
        layout.addWidget(start_line_snd_label)
        layout.addLayout(start_line_snd_x_layout)
        layout.addLayout(start_line_snd_y_layout)

        finish_line_label = QLabel('Finish line coords')
        finish_line_fst_label = QLabel('Start point')
        self._finish_line_a_x_slider, finish_line_fst_x_layout = self._create_slider('X: ')
        self._finish_line_a_y_slider, finish_line_fst_y_layout = self._create_slider('Y: ')
        finish_line_snd_label = QLabel('End point')
        self._finish_line_b_x_slider, finish_line_snd_x_layout = self._create_slider('X: ')
        self._finish_line_b_y_slider, finish_line_snd_y_layout = self._create_slider('Y: ')

        layout.addWidget(finish_line_label, alignment=Qt.AlignCenter)
        layout.addWidget(finish_line_fst_label)
        layout.addLayout(finish_line_fst_x_layout)
        layout.addLayout(finish_line_fst_y_layout)
        layout.addWidget(finish_line_snd_label)
        layout.addLayout(finish_line_snd_x_layout)
        layout.addLayout(finish_line_snd_y_layout)

        self._distance_textbox, distance_layout = self._create_textbox('Distance, m: ')
        layout.addLayout(distance_layout)

        self._start_line_a_x_slider.valueChanged.connect(lambda: self._update_field())
        self._start_line_a_y_slider.valueChanged.connect(lambda: self._update_field())
        self._start_line_b_x_slider.valueChanged.connect(lambda: self._update_field())
        self._start_line_b_y_slider.valueChanged.connect(lambda: self._update_field())
        self._finish_line_a_x_slider.valueChanged.connect(lambda: self._update_field())
        self._finish_line_a_y_slider.valueChanged.connect(lambda: self._update_field())
        self._finish_line_b_x_slider.valueChanged.connect(lambda: self._update_field())
        self._finish_line_b_y_slider.valueChanged.connect(lambda: self._update_field())

        return layout

    @staticmethod
    def _create_slider(description: str) -> Tuple[QSlider, QLayout]:
        layout = QHBoxLayout()

        label = QLabel(description)
        slider = QSlider(Qt.Orientation.Horizontal)

        layout.addWidget(label)
        layout.addWidget(slider)

        return slider, layout

    @staticmethod
    def _create_textbox(description: str) -> Tuple[QLineEdit, QLayout]:
        layout = QHBoxLayout()

        label = QLabel(description)
        textbox = QLineEdit(str(const.DEFAULT_DISTANCE))

        layout.addWidget(label)
        layout.addWidget(textbox)

        return textbox, layout

    def set_capture(self, capture_source: Union[Type[CaptureSource], CaptureSourceImpl]):
        if self._capture is not None:
            self._capture.release()

        self._last_frame_vehicles.clear()

        self._capture = capture_source
        self._set_frame_size(self._capture.frame_size())

    def _set_video_capture(self):
        file_name = QFileDialog.getOpenFileName(
                self._window,
                'Choose a video to capture from...',
                const.PROJECT_PATH,
                'Videos (*.mp4)'
                )[0]
        if file_name:
            self.set_capture(Video(file_name))

    def _set_frame_size(self, frame: FrameSize):
        self._start_line_a_x_slider.setRange(0, frame.width)
        self._start_line_a_y_slider.setRange(0, frame.height)
        self._start_line_b_x_slider.setRange(0, frame.width)
        self._start_line_b_y_slider.setRange(0, frame.height)
        self._finish_line_a_x_slider.setRange(0, frame.width)
        self._finish_line_a_y_slider.setRange(0, frame.height)
        self._finish_line_b_x_slider.setRange(0, frame.width)
        self._finish_line_b_y_slider.setRange(0, frame.height)

        self._start_line_a_x_slider.setValue(frame.width * 0.1)
        self._start_line_a_y_slider.setValue(frame.height * 0.1)
        self._start_line_b_x_slider.setValue(frame.width * 0.9)
        self._start_line_b_y_slider.setValue(frame.height * 0.1)
        self._finish_line_a_x_slider.setValue(frame.width * 0.1)
        self._finish_line_a_y_slider.setValue(frame.height * 0.9)
        self._finish_line_b_x_slider.setValue(frame.width * 0.9)
        self._finish_line_b_y_slider.setValue(frame.height * 0.9)

    def _update_field(self):
        self._field = Trapezoid(
                major=Line(
                        a=Point(
                                x=self._start_line_a_x_slider.value(),
                                y=self._start_line_a_y_slider.value(),
                                ),
                        b=Point(
                                x=self._start_line_b_x_slider.value(),
                                y=self._start_line_b_y_slider.value(),
                                ),
                        ),
                minor=Line(
                        a=Point(
                                x=self._finish_line_a_x_slider.value(),
                                y=self._finish_line_a_y_slider.value(),
                                ),
                        b=Point(
                                x=self._finish_line_b_x_slider.value(),
                                y=self._finish_line_b_y_slider.value(),
                                ),
                        ),
                )

        max_y = self._finish_line_a_y_slider.maximum()

        self._field_points = np.array(
                [[
                    [self._field.major.a.x, max_y - self._field.major.a.y],
                    [self._field.major.b.x, max_y - self._field.major.b.y],
                    [self._field.minor.b.x, max_y - self._field.minor.b.y],
                    [self._field.minor.a.x, max_y - self._field.minor.a.y],
                    ]], dtype='int32'
                )

        field_mask = np.zeros((self._capture.frame_size().height, self._capture.frame_size().width, 1), dtype='int8')
        field_mask = cv2.fillConvexPoly(field_mask, self._field_points, (255,))
        field_mask = np.reshape(field_mask, (field_mask.shape[0], field_mask.shape[1]))
        self._field_mask = np.where(field_mask > 0, True, False).astype(np.bool)

    def _update_traffic_load(self):
        self._traffic_load_text.setText(f'Traffic load: {self._traffic_load:.0%}')

    def process(self):
        if self._capture is None:
            return

        try:
            self._capture.skip_frames(self._calculate_skip_frames())

            frame = self._capture.read()
            vehicles = self._detector.detect(frame)

            # Try to guess which car on previous frame corresponds to the detected car on current frame
            timed_vehicles: List[TimedVehicle] = []
            if len(vehicles) >= len(self._last_frame_vehicles):
                # If current count of cars is bigger than previous cars then
                # we try to make correspondance for all previous cars.
                # For all the rest cars we handle them as they are new
                last_frame_vehicles_count = len(self._last_frame_vehicles)
                for i in range(last_frame_vehicles_count):
                    guessed_vehicle = self._guess_vehicle_on_last_frame(vehicles[i])
                    guessed_vehicle.vehicle = vehicles[i]
                    timed_vehicles.append(guessed_vehicle)
                for i in range(last_frame_vehicles_count, len(vehicles)):
                    timed_vehicles.append(TimedVehicle(vehicles[i]))
            else:
                for vehicle in vehicles:
                    guessed_vehicle = self._guess_vehicle_on_last_frame(vehicle)
                    guessed_vehicle.vehicle = vehicle
                    timed_vehicles.append(guessed_vehicle)

            del vehicles

            frame = self._draw_measurement_field(frame)

            if len(timed_vehicles) > 0:
                combined_vehicle_mask = self._combine_vehicle_masks(timed_vehicles)
                if self._debug_mode:
                    frame = self._draw_mask(combined_vehicle_mask, frame, const.CAR_MASK_COLOR)
                self._traffic_load = self._calculate_traffic_load(combined_vehicle_mask)
            else:
                self._traffic_load = 0.0

            self._update_traffic_load_signal.emit()

            for vehicle in timed_vehicles:
                vehicle_in_field = self._is_point_on_field(vehicle.vehicle.centroid())
                if not vehicle.has_entered_field():
                    if vehicle_in_field:
                        vehicle.field_entered_time = datetime.datetime.now()
                elif not vehicle.has_exited_field():
                    if not vehicle_in_field:
                        vehicle.field_exited_time = datetime.datetime.now()

                if self._debug_mode:
                    vehicle.trace.append(vehicle.vehicle.centroid())

                    if len(vehicle.trace) >= 2:
                        for i in range(1, len(vehicle.trace)):
                            cv2.circle(
                                    frame,
                                    (int(vehicle.vehicle.centroid().x), int(vehicle.vehicle.centroid().y)),
                                    min(self._capture.frame_size().width, self._capture.frame_size().height) // 8,
                                    const.COLOR_WHITE
                                    )

                            cv2.line(
                                    frame,
                                    (int(vehicle.trace[i - 1].x), int(vehicle.trace[i - 1].y)),
                                    (int(vehicle.trace[i].x), int(vehicle.trace[i].y)),
                                    vehicle.vehicle.color()
                                    )

                text = f'{vehicle.vehicle.caption()}'
                if vehicle.has_exited_field():
                    text = f'{text}: {vehicle.speed(int(self._distance_textbox.text())):.0f} km/h'

                circle_coords = (int(vehicle.vehicle.centroid().x), int(vehicle.vehicle.centroid().y))
                text_coords = (int(vehicle.vehicle.centroid().x) - 2 * radius,
                               int(vehicle.vehicle.centroid().y) - 2 * radius)

                cv2.circle(frame, circle_coords, radius + 2, const.COLOR_BLACK, cv2.FILLED)
                if vehicle_in_field:
                    cv2.circle(frame, circle_coords, radius, vehicle.vehicle.color(), cv2.FILLED)

                cv2.putText(frame, text, text_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, const.COLOR_BLACK, 5)
                cv2.putText(frame, text, text_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, const.COLOR_WHITE, 1)

            self._last_frame_vehicles = timed_vehicles

            del timed_vehicles

            image = qimage2ndarray.array2qimage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._pixmap = QPixmap.fromImage(image)
            self._update_pixmap_signal.emit()
        except Exception:
            pass

    def _guess_vehicle_on_last_frame(self, vehicle: VehicleImpl) -> TimedVehicle:
        guessed_vehicle = self._last_frame_vehicles[0]
        min_distance = l2(vehicle.centroid(), guessed_vehicle.vehicle.centroid())
        index = 0

        for i in range(1, len(self._last_frame_vehicles)):
            distance = l2(vehicle.centroid(), self._last_frame_vehicles[i].vehicle.centroid())
            if distance < min_distance:
                guessed_vehicle = self._last_frame_vehicles[i]
                min_distance = distance
                index = i

        if min_distance > (min(self._capture.frame_size().width, self._capture.frame_size().height) / 8):
            guessed_vehicle = TimedVehicle(vehicle)
        else:
            self._last_frame_vehicles.pop(index)

        return guessed_vehicle

    def _is_point_on_field(self, point: Point):
        height = self._capture.frame_size().height
        x, y = int(point.x), int(point.y)

        return self._field_mask[y][x]

    def _draw_measurement_field(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()

        overlay = cv2.fillConvexPoly(overlay, self._field_points, const.COLOR_GREEN)

        overlay = cv2.addWeighted(overlay, const.FIELD_OPACITY, frame, 1.0 - const.FIELD_OPACITY, 0)

        overlay = self._draw_line(overlay, self._field.major, const.COLOR_GREEN)
        overlay = self._draw_line(overlay, self._field.minor, const.COLOR_RED)
        overlay = self._draw_point(overlay, self._field.major.a, const.COLOR_GREEN)
        overlay = self._draw_point(overlay, self._field.major.b, const.COLOR_RED)
        overlay = self._draw_point(overlay, self._field.minor.a, const.COLOR_GREEN)
        overlay = self._draw_point(overlay, self._field.minor.b, const.COLOR_RED)

        return cv2.addWeighted(overlay, const.FIELD_OPACITY, frame, 1.0 - const.FIELD_OPACITY, 0)

    def _draw_line(self, img, line: Line, color: Tuple[int, int, int]):
        max_y = self._capture.frame_size().height - 1

        return cv2.line(
                img,
                pt1=(line.a.x, max_y - line.a.y),
                pt2=(line.b.x, max_y - line.b.y),
                color=color,
                thickness=2,
                )

    def _draw_point(self, img, point: Point, color: Tuple[int, int, int]):
        max_y = self._capture.frame_size().height - 1

        return cv2.circle(
                img,
                center=(point.x, max_y - point.y),
                radius=10,
                color=color,
                )

    @staticmethod
    def _combine_vehicle_masks(vehicles: List[TimedVehicle]) -> np.ndarray:
        mask = vehicles[0].vehicle.mask().copy()

        for i in range(1, len(vehicles)):
            mask |= vehicles[i].vehicle.mask()

        return mask

    @staticmethod
    def _draw_mask(mask: np.ndarray, frame: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        for c in range(frame.shape[2]):
            frame[:, :, c] = np.where(
                    mask > 0,
                    frame[:, :, c] * (1 - const.FIELD_OPACITY) + color[c] * const.FIELD_OPACITY,
                    frame[:, :, c]
                    )

        return frame

    def _update_pixmap(self):
        if self._pixmap is not None:
            self._video.setPixmap(self._pixmap)

    def _calculate_skip_frames(self) -> int:
        current_time = datetime.datetime.now()
        frames = 0

        if self._last_frame_time is not None:
            time_delta = current_time - self._last_frame_time
            frames = round((self._capture.fps() * time_delta.microseconds) / 1_000_000)

        self._last_frame_time = current_time

        return frames

    def _calculate_traffic_load(self, vehicle_mask):
        vehicle_on_field_mask = self._field_mask & vehicle_mask
        vehicle_pixels = np.count_nonzero(vehicle_on_field_mask)
        field_pixels = np.count_nonzero(self._field_mask)

        return vehicle_pixels / field_pixels

    def run(self):
        self._update_traffic_load_signal.connect(self._update_traffic_load)
        self._update_pixmap_signal.connect(self._update_pixmap)

        processing = VideoProcessing(self)
        processing.setAutoDelete(True)

        pool = QThreadPool.globalInstance()

        self._window.show()
        pool.start(processing)
        return self.exec_()


class VideoProcessing(QRunnable):
    def __init__(self, app: App):
        super().__init__()
        self._app = app

    def run(self):
        while True:
            self._app.process()


def l2(centroid1: Point, centroid2: Point) -> float:
    return math.sqrt(math.pow(centroid1.x - centroid2.x, 2.0) + math.pow(centroid1.y - centroid2.y, 2.0))
