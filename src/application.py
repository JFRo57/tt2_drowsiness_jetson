import os
import time
import traceback

import cv2

from .alert_controller import AlertController
from .calibration import CalibrationManager
from .camera import CameraManager
from .event_logger import EventLogger
from .face_analyzer import FaceAnalyzer
from .fatigue_detector import FatigueDetector
from .gpio_controller import GPIOController
from .mode_controller import ModeController
from .presentation_ui import PresentationUI
from .shutdown_manager import ShutdownManager


def default_config():
    return {
        "camera": {"sensor_id": 0, "capture_width": 1280, "capture_height": 720, "processing_width": 640, "processing_height": 360, "display_width": 1100, "display_height": 620, "fps": 30, "flip_method": 2, "reconnect_attempts": 3},
        "dlib": {"predictor_path": "models/shape_predictor_68_face_landmarks.dat", "upsample": 0, "detection_interval_frames": 5, "use_correlation_tracker": True},
        "preprocessing": {"use_clahe": True, "clahe_clip_limit": 2.0, "clahe_grid_size": 8, "use_gamma": False, "gamma": 1.0},
        "fatigue": {"ear_threshold": 0.22, "use_session_calibration": True, "blink_min_seconds": 0.08, "blink_max_seconds": 0.70, "prealert_closed_seconds": 0.45, "alert_closed_seconds": 0.85, "critical_closed_seconds": 1.6, "recovery_seconds": 1.0, "perclos_window_seconds": 60, "perclos_warning_threshold": 0.25, "perclos_alert_threshold": 0.35, "mar_threshold": 0.65, "yawn_min_seconds": 1.0, "head_nod_pitch_threshold": 18.0, "head_nod_min_seconds": 0.8, "gaze_away_warning_seconds": 2.0, "no_face_warning_seconds": 2.0},
        "calibration": {"duration_seconds": 4.0, "min_samples": 12, "quality_threshold": 0.30, "min_ear_gap": 0.015, "max_ear_std": 0.08, "max_profile_distance": 4.0, "profile_min_confidence": 0.15, "profile_warning_seconds": 0.8, "feature_scales": {"ear": 0.04, "mar": 0.15, "pitch": 12.0, "yaw": 15.0, "roll": 15.0}},
        "gpio": {"enabled": False, "simulation_mode": True, "numbering": "BOARD"},
        "buzzer": {"enabled": True, "type": "pwm_native", "board_pin": 33, "active_high": False, "muted": False, "idle_frequency": 3000, "pwm_duty_cycle": 50, "pwm_chip": 0, "pwm_channel": 2, "min_frequency": 2000, "max_frequency": 5000},
        "leds": {"enabled": True, "active_high": True, "green_board_pin": 29, "yellow_board_pin": 31, "red_board_pin": 32},
        "switch": {"enabled": False, "topology": "on_off_on", "auto_board_pin": 35, "emergency_board_pin": 37, "center_mode": "MAINTENANCE", "active_low": True, "debounce_ms": 100},
        "interface": {"presentation_enabled": True, "show_landmarks": True, "show_information_panel": True, "show_virtual_leds": True, "close_action": "FULL_SHUTDOWN"},
        "logging": {"enabled": False, "events_only": True, "path": "logs/events.csv", "save_images": False, "record_video": False},
    }


def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


class DrowsinessApplication(object):
    def __init__(self, config, simulation=False):
        self.config = config
        self.simulation = simulation
        self.shutdown = ShutdownManager()
        self.camera = CameraManager(config["camera"])
        self.gpio = GPIOController(config, force_simulation=simulation)
        self.mode = ModeController()
        self.alerts = AlertController(self.gpio)
        self.detector = FatigueDetector(config)
        self.calibration = CalibrationManager(config)
        self.logger = EventLogger(config)
        self.ui = PresentationUI(config) if config["interface"].get("presentation_enabled", True) else None
        self.face = None
        self.analysis_fps = 0.0
        self.analysis_ms = 0.0
        self.started_at = time.monotonic()
        self.error = None
        self.forced_test_state = None

    def start_calibration(self, profile="OPEN"):
        self.forced_test_state = None
        self.detector.reset()
        self.calibration.start(profile)

    def run(self):
        exit_code = 0
        try:
            self.gpio.setup()
            self.logger.open()
            self.face = FaceAnalyzer(self.config)
            if not self.camera.start():
                raise RuntimeError(self.camera.error or "No se pudo iniciar la camara")
            if self.ui:
                self.ui.create()
            self.detector.state = "NORMAL"
            self._loop()
        except Exception as exc:
            exit_code = 1
            self.error = str(exc)
            print("ERROR:", exc)
            traceback.print_exc()
            self.detector.state = "ERROR"
            self.alerts.update("ERROR", self.mode.mode)
        finally:
            self.cleanup()
        return exit_code

    def _loop(self):
        frames = 0
        last_fps = time.monotonic()
        last_metrics = {"face_detected": False, "quality": 0.0}
        while not self.shutdown.requested:
            self.mode.update_from_switch(self.gpio)
            if self.mode.mode == ModeController.EMERGENCY:
                state, reason = self.detector.update({"face_detected": False}, self.mode.mode)
                self.alerts.update(state, self.mode.mode)
                self._render_if_needed(last_metrics)
                if not self.ui:
                    time.sleep(0.05)
                continue
            frame, ts = self.camera.get_latest_frame()
            if frame is None:
                if self.camera.error:
                    raise RuntimeError(self.camera.error)
                time.sleep(0.01)
                continue
            if self.ui and self.ui.paused:
                self._render_if_needed(last_metrics, frame)
                continue
            if self.forced_test_state:
                metrics = dict(last_metrics)
                metrics["runtime_seconds"] = time.monotonic() - self.started_at
                metrics["analysis_ms"] = self.analysis_ms
                state, reason = self._apply_forced_test_state()
                self.alerts.update(state, self.mode.mode)
                self.logger.log_transition(self.mode.mode, self.detector.previous_state, state, reason, metrics, self.analysis_fps, self.gpio)
                self._render_if_needed(metrics, frame)
                if self.camera.error:
                    raise RuntimeError(self.camera.error)
                continue
            analysis_started = time.monotonic()
            metrics = self.face.analyze(frame)
            classification = self.calibration.classify(metrics)
            metrics["calibrated_profile"] = classification["profile"]
            metrics["calibrated_profile_confidence"] = classification["confidence"]
            metrics["calibrated_profile_scores"] = classification["scores"]
            metrics.update(self.calibration.status_metrics())
            last_metrics = metrics
            was_calibrating = self.calibration.active
            calibration_result = self.calibration.update(metrics)
            if calibration_result is not None:
                self.detector.apply_calibration(calibration_result)
                metrics.update(self.calibration.status_metrics())
            if was_calibrating and not self.calibration.active:
                self.detector.reset()
            if self.calibration.active:
                self.detector.previous_state = self.detector.state
                self.detector.state = "CALIBRANDO"
                self.detector.reason = self.calibration.message
                state, reason = self.detector.state, self.detector.reason
            else:
                state, reason = self.detector.update(metrics, self.mode.mode)
            self.analysis_ms = (time.monotonic() - analysis_started) * 1000.0
            metrics["runtime_seconds"] = time.monotonic() - self.started_at
            metrics["analysis_ms"] = self.analysis_ms
            self.alerts.update(state, self.mode.mode)
            self.logger.log_transition(self.mode.mode, self.detector.previous_state, state, reason, metrics, self.analysis_fps, self.gpio)
            frames += 1
            now = time.monotonic()
            if now - last_fps >= 1.0:
                self.analysis_fps = frames / (now - last_fps)
                frames = 0
                last_fps = now
            self._render_if_needed(metrics, frame)
            if self.camera.error:
                raise RuntimeError(self.camera.error)

    def _render_if_needed(self, metrics, frame=None):
        if not self.ui:
            time.sleep(0.001)
            return
        if frame is None:
            frame, _ = self.camera.get_latest_frame()
        if frame is not None:
            metrics = dict(metrics)
            metrics.update(self.calibration.status_metrics())
            metrics.setdefault("runtime_seconds", time.monotonic() - self.started_at)
            metrics.setdefault("analysis_ms", self.analysis_ms)
            self.ui.draw(frame, metrics, self.detector, self.mode, self.gpio, self.camera.capture_fps, self.analysis_fps, self.hardware_mode())
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            self.ui.handle_key(key, self)
        if self.ui.is_closed():
            self.shutdown.request("Ventana cerrada")

    def hardware_mode(self):
        return "GPIO simulado" if self.gpio.simulation_mode else "GPIO fisico"

    def set_forced_test_state(self, state):
        if state is None:
            self.forced_test_state = None
            self.detector.reset()
            return
        if state in self.detector.STATES:
            self.forced_test_state = state

    def _apply_forced_test_state(self):
        now = time.monotonic()
        self.detector.previous_state = self.detector.state
        state = self.forced_test_state
        reasons = {
            "NORMAL": "Prueba manual de salidas: NORMAL",
            "POSIBLE_SOMNOLENCIA": "Prueba manual de salidas: POSIBLE_SOMNOLENCIA",
            "ALERTA": "Prueba manual de salidas: ALERTA",
            "ALERTA_CRITICA": "Prueba manual de salidas: ALERTA_CRITICA",
            "ROSTRO_NO_DETECTADO": "Prueba manual de salidas: ROSTRO_NO_DETECTADO",
        }
        return self.detector._set(state, reasons.get(state, "Prueba manual de salidas"), now)

    def cleanup(self):
        try:
            self.gpio.all_outputs_off()
        except Exception:
            pass
        try:
            self.camera.stop()
        except Exception:
            pass
        try:
            self.logger.close()
        except Exception:
            pass
        try:
            self.gpio.cleanup()
        except Exception:
            pass
        try:
            if self.ui:
                self.ui.close()
            else:
                cv2.destroyAllWindows()
        except Exception:
            pass
