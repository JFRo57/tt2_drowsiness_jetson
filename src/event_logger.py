import csv
import os
import time


class EventLogger(object):
    FIELDS = [
        "timestamp", "mode", "previous_state", "new_state", "reason", "ear", "closed_seconds",
        "perclos", "mar", "possible_yawn", "pitch", "yaw", "roll", "gaze",
        "face_detected", "fps", "buzzer", "buzzer_frequency", "led_active"
    ]

    def __init__(self, config):
        self.config = config["logging"]
        self.enabled = bool(self.config.get("enabled", False))
        self.file = None
        self.writer = None
        self.error = None

    def open(self):
        if not self.enabled:
            return
        path = self.config.get("path", "logs/events.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        if self.file.tell() == 0:
            self.writer.writeheader()

    def log_transition(self, mode, previous, new, reason, metrics, fps, gpio):
        if not self.enabled or self.writer is None or previous == new:
            return
        try:
            leds = ",".join([k for k, v in gpio.leds.items() if v])
            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode,
                "previous_state": previous,
                "new_state": new,
                "reason": reason,
                "ear": metrics.get("ear"),
                "closed_seconds": metrics.get("closed_seconds"),
                "perclos": metrics.get("perclos"),
                "mar": metrics.get("mar"),
                "possible_yawn": metrics.get("possible_yawn"),
                "pitch": metrics.get("pitch"),
                "yaw": metrics.get("yaw"),
                "roll": metrics.get("roll"),
                "gaze": metrics.get("gaze"),
                "face_detected": metrics.get("face_detected"),
                "fps": fps,
                "buzzer": gpio.buzzer_on,
                "buzzer_frequency": getattr(gpio, "buzzer_frequency", 0),
                "led_active": leds,
            }
            self.writer.writerow(row)
            self.file.flush()
        except Exception as exc:
            self.error = "Error de escritura del log: %s" % exc

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
