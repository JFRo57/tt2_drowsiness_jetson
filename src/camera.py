import threading
import time

import cv2


class CameraManager(object):
    def __init__(self, config):
        self.config = config
        self.cap = None
        self.thread = None
        self.lock = threading.Lock()
        self.running = False
        self.latest_frame = None
        self.latest_time = 0.0
        self.capture_fps = 0.0
        self.error = None
        self.empty_frames = 0

    def build_pipeline(self):
        c = self.config
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
            % (
                int(c["sensor_id"]),
                int(c["capture_width"]),
                int(c["capture_height"]),
                int(c["fps"]),
                int(c["flip_method"]),
                int(c["processing_width"]),
                int(c["processing_height"]),
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
        with self.lock:
            self.latest_frame = frame
            self.latest_time = time.monotonic()
        self.error = None
        return True

    def start(self):
        if not self.open():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._loop)
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
            with self.lock:
                self.latest_frame = frame
                self.latest_time = now
            frames += 1
            if now - last >= 1.0:
                self.capture_fps = frames / (now - last)
                frames = 0
                last = now

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), self.latest_time

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(2.0)
            self.thread = None
        self.release()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
