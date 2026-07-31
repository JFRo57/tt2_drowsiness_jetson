import csv
import os
import time


class EventLogger(object):
    """Registro de transiciones y diagnostico cuadro a cuadro de la vision."""

    FIELDS = [
        "timestamp", "monotonic_seconds", "record_type", "mode",
        "previous_state", "new_state", "reason", "consistency_warning",
        "calibrated_profile", "profile_confidence", "profile_seconds",
        "eye_event_state", "eye_measurement_valid", "valid_eye_count",
        "eye_measurement_partial", "left_eye_reliable", "right_eye_reliable",
        "eye_quality", "left_eye_sharpness", "right_eye_sharpness",
        "eye_ear_difference", "ear_left", "ear_right", "ear_raw", "ear",
        "closure_left", "closure_right", "closure_normalized",
        "deep_closure_active", "closed_seconds", "strong_elapsed_seconds",
        "events_active", "new_events", "weak_evidence", "strong_evidence",
        "critical_evidence", "baseline_blink_seconds", "last_blink_seconds",
        "perclos", "perclos_valid_seconds", "perclos_closed_seconds",
        "perclos_total_seconds", "perclos_coverage", "perclos_reliable",
        "recent_yawns", "recent_nods", "mar", "pitch", "yaw", "roll",
        "pose_reliable", "vision_state", "vision_reason", "face_detected",
        "quality", "brightness", "analysis_source", "frame_sequence", "fps",
        "analysis_ms", "frame_age_ms", "alert", "buzzer",
        "buzzer_frequency", "led_active",
    ]

    def __init__(self, config, clock=None):
        self.config = config["logging"]
        self.enabled = bool(self.config.get("enabled", False))
        self.events_only = bool(self.config.get("events_only", True))
        self.snapshot_interval = float(
            self.config.get("snapshot_interval_seconds", 5.0))
        self.diagnostic_interval = float(
            self.config.get("diagnostic_interval_seconds", 0.5))
        self.flush_interval = float(self.config.get("flush_interval_seconds", 2.0))
        self.clock = clock or time.monotonic
        self.file = None
        self.writer = None
        self.error = None
        self.last_snapshot = None
        self.last_flush = None
        self.last_vision_state = None
        self.last_profile = None
        self.last_eye_valid = None
        self.last_left_reliable = None
        self.last_right_reliable = None
        self.last_warning = None
        self.path = None

    def open(self):
        if not self.enabled:
            return
        path = self.config.get("path", "logs/events.csv")
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        path = self._compatible_path(path)
        self.path = path
        self.file = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        if self.file.tell() == 0:
            self.writer.writeheader()
        now = self.clock()
        self.last_snapshot = now
        self.last_flush = now

    def _compatible_path(self, path):
        expected = ",".join(self.FIELDS)
        candidate = path
        version = 2
        while os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            with open(candidate, "r") as existing:
                if existing.readline().strip() == expected:
                    return candidate
            root, extension = os.path.splitext(path)
            candidate = "%s_v%d%s" % (root, version, extension or ".csv")
            version += 1
        return candidate

    def log_transition(self, mode, previous, new, reason, metrics, fps, gpio):
        if not self.enabled or self.writer is None:
            return
        now = self.clock()
        vision_state = metrics.get("vision_state")
        profile = metrics.get("calibrated_profile")
        eye_valid = bool(metrics.get("eye_measurement_valid", False))
        left_reliable = bool(metrics.get("left_eye_reliable", False))
        right_reliable = bool(metrics.get("right_eye_reliable", False))
        warning = self.consistency_warning(new, metrics)
        state_changed = previous != new
        signal_changed = bool(
            vision_state != self.last_vision_state
            or profile != self.last_profile
            or eye_valid != self.last_eye_valid
            or left_reliable != self.last_left_reliable
            or right_reliable != self.last_right_reliable
            or warning != self.last_warning
        )
        has_event = bool(metrics.get("new_events") or metrics.get("critical_evidence"))
        interval = self.diagnostic_interval if not self.events_only else self.snapshot_interval
        periodic = bool(
            not self.events_only
            and (self.last_snapshot is None or now - self.last_snapshot >= interval)
        )
        if not state_changed and not signal_changed and not has_event and not periodic:
            return
        record_type = self._record_type(state_changed, signal_changed, has_event,
                                        periodic, warning)
        try:
            leds = ",".join(key for key, value in gpio.leds.items() if value)
            strong_since = metrics.get("strong_since")
            strong_elapsed = (max(0.0, now - float(strong_since))
                              if strong_since is not None else 0.0)
            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "monotonic_seconds": "%.6f" % now,
                "record_type": record_type, "mode": mode,
                "previous_state": previous, "new_state": new, "reason": reason,
                "consistency_warning": warning,
                "calibrated_profile": profile,
                "profile_confidence": metrics.get("calibrated_profile_confidence"),
                "profile_seconds": metrics.get("calibrated_profile_seconds"),
                "eye_event_state": metrics.get("eye_event_state"),
                "eye_measurement_valid": eye_valid,
                "valid_eye_count": metrics.get("valid_eye_count"),
                "eye_measurement_partial": metrics.get("eye_measurement_partial"),
                "left_eye_reliable": left_reliable,
                "right_eye_reliable": right_reliable,
                "eye_quality": metrics.get("eye_quality"),
                "left_eye_sharpness": metrics.get("left_eye_sharpness"),
                "right_eye_sharpness": metrics.get("right_eye_sharpness"),
                "eye_ear_difference": metrics.get("eye_ear_difference"),
                "ear_left": metrics.get("left_ear"),
                "ear_right": metrics.get("right_ear"),
                "ear_raw": metrics.get("ear_raw"), "ear": metrics.get("ear"),
                "closure_left": metrics.get("left_closure_normalized"),
                "closure_right": metrics.get("right_closure_normalized"),
                "closure_normalized": metrics.get("closure_normalized"),
                "deep_closure_active": metrics.get("deep_closure_active"),
                "closed_seconds": metrics.get("closed_seconds"),
                "strong_elapsed_seconds": strong_elapsed,
                "events_active": self._join(metrics.get("active_events")),
                "new_events": self._join(metrics.get("new_events")),
                "weak_evidence": self._join(metrics.get("weak_evidence")),
                "strong_evidence": self._join(metrics.get("strong_evidence")),
                "critical_evidence": self._join(metrics.get("critical_evidence")),
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
                "mar": metrics.get("mar"), "pitch": metrics.get("pitch"),
                "yaw": metrics.get("yaw"), "roll": metrics.get("roll"),
                "pose_reliable": metrics.get("pose_reliable"),
                "vision_state": vision_state,
                "vision_reason": metrics.get("vision_reason"),
                "face_detected": metrics.get("face_detected"),
                "quality": metrics.get("quality"),
                "brightness": metrics.get("brightness"),
                "analysis_source": metrics.get("analysis_source"),
                "frame_sequence": metrics.get("frame_sequence"), "fps": fps,
                "analysis_ms": metrics.get("analysis_ms"),
                "frame_age_ms": metrics.get("frame_age_ms"),
                "alert": metrics.get("active_alert"), "buzzer": gpio.buzzer_on,
                "buzzer_frequency": getattr(gpio, "buzzer_frequency", 0),
                "led_active": leds,
            }
            self.writer.writerow(row)
            self.last_snapshot = now
            self.last_vision_state = vision_state
            self.last_profile = profile
            self.last_eye_valid = eye_valid
            self.last_left_reliable = left_reliable
            self.last_right_reliable = right_reliable
            self.last_warning = warning
            immediate = bool(state_changed or warning or has_event
                             or vision_state == "CAMARA_OBSTRUIDA_O_FALLO")
            if immediate or self.last_flush is None or now - self.last_flush >= self.flush_interval:
                self.file.flush()
                self.last_flush = now
        except Exception as exc:
            self.error = "Error de escritura del log: %s" % exc

    @staticmethod
    def consistency_warning(state, metrics):
        if not metrics.get("eye_measurement_valid", False):
            return "MEDICION_OCULAR_INVALIDA"
        profile = metrics.get("calibrated_profile")
        deep = bool(metrics.get("deep_closure_active", False))
        profile_seconds = float(metrics.get("calibrated_profile_seconds", 0.0) or 0.0)
        if profile == "CLOSED" and not deep:
            return "PERFIL_CERRADO_SIN_CIERRE_PROFUNDO"
        if profile == "CLOSED" and state == "ALERTA" and profile_seconds >= 0.8:
            return "OJOS_CERRADOS_ESTADO_ALERTA"
        if profile == "OPEN" and state in ("SOMNOLENCIA", "CRITICO") and profile_seconds >= 0.8:
            return "ESTADO_RETENIDO_CON_OJOS_ABIERTOS"
        return ""

    @staticmethod
    def _record_type(state_changed, signal_changed, has_event, periodic, warning):
        if warning:
            return "INCONSISTENCIA"
        if state_changed:
            return "TRANSICION"
        if has_event:
            return "EVENTO"
        if signal_changed:
            return "CAMBIO_VISION"
        return "MUESTRA" if periodic else "DIAGNOSTICO"

    @staticmethod
    def _join(values):
        return "|".join(values or [])

    def close(self):
        if self.file is not None:
            try:
                self.file.flush()
            finally:
                self.file.close()
                self.file = None
