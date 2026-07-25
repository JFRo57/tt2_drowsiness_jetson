import csv
import os
import time


class EventLogger(object):
    FIELDS = [
        "timestamp", "monotonic_seconds", "mode", "previous_state", "new_state",
        "reason", "events_active", "new_events", "ear_left", "ear_right", "ear",
        "closure_left", "closure_right", "closure_normalized", "closed_seconds",
        "baseline_blink_seconds", "last_blink_seconds", "perclos",
        "perclos_valid_seconds", "perclos_closed_seconds", "perclos_total_seconds",
        "perclos_coverage", "perclos_reliable", "recent_yawns", "recent_nods",
        "mar", "pitch", "yaw", "roll",
        "vision_state", "vision_reason", "face_detected", "fps", "alert",
        "buzzer", "buzzer_frequency", "led_active",
    ]

    def __init__(self, config, clock=None):
        self.config = config["logging"]
        self.enabled = bool(self.config.get("enabled", False))
        self.events_only = bool(self.config.get("events_only", True))
        self.snapshot_interval = float(
            self.config.get("snapshot_interval_seconds", 5.0)
        )
        self.flush_interval = float(self.config.get("flush_interval_seconds", 2.0))
        self.clock = clock or time.monotonic
        self.file = None
        self.writer = None
        self.error = None
        self.last_snapshot = None
        self.last_flush = None
        self.last_vision_state = None
        self.path = None

    def open(self):
        if not self.enabled:
            return
        path = self.config.get("path", "logs/events.csv")
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        expected = ",".join(self.FIELDS)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r") as existing:
                header = existing.readline().strip()
            if header != expected:
                root, extension = os.path.splitext(path)
                path = root + "_v2" + (extension or ".csv")
        self.path = path
        self.file = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        if self.file.tell() == 0:
            self.writer.writeheader()
        now = self.clock()
        self.last_snapshot = now
        self.last_flush = now

    def log_transition(self, mode, previous, new, reason, metrics, fps, gpio):
        if not self.enabled or self.writer is None:
            return
        now = self.clock()
        vision_state = metrics.get("vision_state")
        state_changed = previous != new
        vision_changed = vision_state != self.last_vision_state
        has_new_event = bool(metrics.get("new_events"))
        critical_event = bool(metrics.get("critical_evidence"))
        periodic = bool(
            not self.events_only
            and (self.last_snapshot is None
                 or now - self.last_snapshot >= self.snapshot_interval)
        )
        if not state_changed and not vision_changed and not has_new_event and not periodic:
            return
        try:
            leds = ",".join([key for key, value in gpio.leds.items() if value])
            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "monotonic_seconds": "%.6f" % now,
                "mode": mode,
                "previous_state": previous,
                "new_state": new,
                "reason": reason,
                "events_active": "|".join(metrics.get("active_events", [])),
                "new_events": "|".join(metrics.get("new_events", [])),
                "ear_left": metrics.get("left_ear"),
                "ear_right": metrics.get("right_ear"),
                "ear": metrics.get("ear"),
                "closure_left": metrics.get("left_closure_normalized"),
                "closure_right": metrics.get("right_closure_normalized"),
                "closure_normalized": metrics.get("closure_normalized"),
                "closed_seconds": metrics.get("closed_seconds"),
                "baseline_blink_seconds": metrics.get("baseline_blink_median_seconds"),
                "last_blink_seconds": metrics.get("last_blink_seconds"),
                "perclos": metrics.get("perclos"),
                "perclos_valid_seconds": metrics.get("perclos_valid_seconds"),
                "perclos_closed_seconds": metrics.get("perclos_closed_seconds"),
                "perclos_total_seconds": metrics.get("perclos_total_seconds"),
                "perclos_coverage": metrics.get("perclos_coverage"),
                "perclos_reliable": metrics.get("perclos_reliable"),
                "recent_yawns": metrics.get("recent_yawns"),
                "recent_nods": metrics.get("recent_nods"),
                "mar": metrics.get("mar"),
                "pitch": metrics.get("pitch"),
                "yaw": metrics.get("yaw"),
                "roll": metrics.get("roll"),
                "vision_state": vision_state,
                "vision_reason": metrics.get("vision_reason"),
                "face_detected": metrics.get("face_detected"),
                "fps": fps,
                "alert": metrics.get("active_alert"),
                "buzzer": gpio.buzzer_on,
                "buzzer_frequency": getattr(gpio, "buzzer_frequency", 0),
                "led_active": leds,
            }
            self.writer.writerow(row)
            self.last_snapshot = now
            self.last_vision_state = vision_state
            immediate = bool(
                new == "CRITICO"
                or critical_event
                or vision_state == "CAMARA_OBSTRUIDA_O_FALLO"
            )
            if immediate or self.last_flush is None or (
                now - self.last_flush >= self.flush_interval
            ):
                self.file.flush()
                self.last_flush = now
        except Exception as exc:
            self.error = "Error de escritura del log: %s" % exc

    def close(self):
        if self.file is not None:
            try:
                self.file.flush()
            finally:
                self.file.close()
                self.file = None
