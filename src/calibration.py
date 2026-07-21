import time

import numpy as np


class CalibrationManager(object):
    def __init__(self, config):
        self.config = config
        self.active = False
        self.started = 0.0
        self.duration = 4.0
        self.samples = []
        self.progress = 0.0
        self.result_threshold = None
        self.message = "Calibracion inactiva"

    def start(self):
        self.active = True
        self.started = time.monotonic()
        self.samples = []
        self.progress = 0.0
        self.result_threshold = None
        self.message = "Mire al frente y mantenga los ojos abiertos"

    def update(self, metrics):
        if not self.active:
            return None
        now = time.monotonic()
        self.progress = min(1.0, (now - self.started) / self.duration)
        if metrics.get("face_detected") and metrics.get("ear") and metrics.get("quality", 0.0) >= 0.35:
            self.samples.append(float(metrics["ear"]))
        if self.progress < 1.0:
            return None
        self.active = False
        if len(self.samples) < 12:
            self.message = "Calibracion invalida: muestras insuficientes"
            return None
        arr = np.array(self.samples)
        med = float(np.median(arr))
        std = float(np.std(arr))
        if med <= 0.15 or std > 0.08:
            self.message = "Calibracion invalida: EAR inestable"
            return None
        self.result_threshold = max(0.15, min(0.32, med * 0.72))
        self.message = "Umbral EAR de sesion: %.3f" % self.result_threshold
        return self.result_threshold
