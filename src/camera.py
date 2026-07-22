import threading
import time

import cv2


class CameraManager(object):
    def __init__(self, config):
        self.config = config
        self.cap = None
        self.thread = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.running = False
        self.latest_frame = None
        self.latest_time = 0.0
        self.frame_sequence = 0
        self.capture_fps = 0.0
        self.error = None
        self.empty_frames = 0

    def build_pipeline(self):
        c = self.config
        output_format = str(c.get("output_format", "BGRx")).upper()
        if output_format == "BGRX":
            conversion = ""
        elif output_format == "BGR":
            conversion = "videoconvert ! video/x-raw, format=(string)BGR ! "
        else:
            raise ValueError("camera.output_format debe ser BGRx o BGR")
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! %sappsink drop=true max-buffers=1 sync=false"
            % (
                int(c["sensor_id"]),
                int(c["capture_width"]),
                int(c["capture_height"]),
                int(c["fps"]),
                int(c["flip_method"]),
                int(c["processing_width"]),
                int(c["processing_height"]),
                conversion,
            )
        )

    def open(self):
        self.release()
        pipeline = self.build_pipeline()
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.error = "No se pudo abrir la camara CSI con nvarguscamerasrc"
            return False
        ok, frame = self.cap.read()
        if not ok or frame is None or frame.size == 0:
            self.release()
            self.error = "La camara abrio, pero no entrego frames validos"
            return False
        with self.condition:
            self.latest_frame = frame
            self.latest_time = time.monotonic()
            self.frame_sequence += 1
            self.condition.notify_all()
        self.error = None
        return True

    def start(self):
        if not self.open():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="camera-capture")
        self.thread.daemon = True
        self.thread.start()
        return True

    def _loop(self):
        last = time.monotonic()
        frames = 0
        attempts = 0
        max_attempts = int(self.config.get("reconnect_attempts", 3))
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if attempts >= max_attempts:
                    self.error = "Fallo de camara: intentos de reconexion agotados"
                    self.running = False
                    with self.condition:
                        self.condition.notify_all()
                    break
                attempts += 1
                time.sleep(0.2)
                self.open()
                continue
            ok, frame = self.cap.read()
            now = time.monotonic()
            if not ok or frame is None or frame.size == 0:
                self.empty_frames += 1
                if self.empty_frames > 15:
                    self.release()
                continue
            self.empty_frames = 0
            attempts = 0
            with self.condition:
                self.latest_frame = frame
                self.latest_time = now
                self.frame_sequence += 1
                self.condition.notify_all()
            frames += 1
            if now - last >= 1.0:
                self.capture_fps = frames / (now - last)
                frames = 0
                last = now

    def get_latest_frame(self, copy=True):
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            frame = self.latest_frame.copy() if copy else self.latest_frame
            return frame, self.latest_time

    def wait_for_frame(self, after_sequence=-1, timeout=0.1, copy=False):
        """Return only a frame newer than the requested sequence.

        Capture replaces the NumPy buffer reference and never mutates a
        published frame, so consumers can safely avoid a full-frame copy.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self.condition:
            while self.running and self.frame_sequence <= after_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None, 0.0, after_sequence
                self.condition.wait(remaining)
            if self.latest_frame is None or self.frame_sequence <= after_sequence:
                return None, 0.0, after_sequence
            frame = self.latest_frame.copy() if copy else self.latest_frame
            return frame, self.latest_time, self.frame_sequence

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(2.0)
            self.thread = None
        self.release()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
