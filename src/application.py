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
        "camera": {"sensor_id": 0, "capture_width": 1280, "capture_height": 720, "processing_width": 640, "processing_height": 360, "display_width": 1100, "display_height": 620, "fps": 30, "flip_method": 2, "output_format": "BGRx", "reconnect_attempts": 3},
        "dlib": {"predictor_path": "models/shape_predictor_68_face_landmarks.dat", "upsample": 0, "detection_interval_frames": 12, "no_face_detection_interval_frames": 3, "detector_scale": 0.5, "tracking_mode": "landmarks", "use_correlation_tracker": False, "tracker_quality_threshold": 6.5, "pose_interval_frames": 2, "gaze_interval_frames": 2},
        "face_detection": {"backend": "auto", "backend_order": ["dlib_cnn_cuda", "opencv_cuda_fp16", "dlib_hog"], "allow_fallback": True, "warmup": True, "confidence_threshold": 0.55, "dlib_cnn_model_path": "models/mmod_human_face_detector.dat", "opencv_config_path": "models/deploy.prototxt", "opencv_model_path": "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"},
        "preprocessing": {"use_clahe": True, "adaptive_clahe": True, "clahe_dark_threshold": 75.0, "clahe_bright_threshold": 205.0, "clahe_clip_limit": 2.0, "clahe_grid_size": 8, "use_gamma": False, "gamma": 1.0},
        "performance": {"target_analysis_fps": 20.0, "frame_wait_timeout_seconds": 0.1, "opencv_threads": 2, "opencv_optimized": True},
        "stability": {"landmark_shape_alpha": 0.30, "landmark_translation_alpha": 0.75, "tracking_rect_alpha": 0.35, "redetection_rect_alpha": 0.30, "redetection_min_iou": 0.15, "redetection_max_center_shift": 0.45, "redetection_miss_tolerance": 2, "ear_median_window": 3, "ear_hysteresis": 0.012, "close_confirm_seconds": 0.08, "open_confirm_seconds": 0.15, "unreliable_hold_seconds": 0.25, "face_loss_hold_seconds": 0.25, "eye_quality_threshold": 0.25, "min_eye_width_pixels": 10.0, "min_eye_sharpness": 12.0, "max_eye_ear_difference": 0.12, "max_eye_ear_difference_ratio": 0.65, "max_eye_yaw_degrees": 32.0},
        "fatigue": {"ear_threshold": 0.22, "use_session_calibration": True, "blink_min_seconds": 0.08, "blink_max_seconds": 0.70, "prealert_closed_seconds": 0.45, "alert_closed_seconds": 0.85, "critical_closed_seconds": 1.6, "recovery_seconds": 1.0, "perclos_window_seconds": 60, "perclos_warning_threshold": 0.25, "perclos_alert_threshold": 0.35, "mar_threshold": 0.65, "yawn_min_seconds": 1.0, "head_nod_pitch_threshold": 18.0, "head_nod_min_seconds": 0.8, "gaze_away_warning_seconds": 2.0, "no_face_warning_seconds": 2.0},
        "calibration": {"duration_seconds": 5.0, "min_samples": 30, "quality_threshold": 0.30, "min_ear_gap": 0.020, "max_ear_std": 0.035, "max_ear_mad": 0.025, "max_profile_overlap_ratio": 0.65, "max_profile_distance": 4.0, "profile_min_confidence": 0.35, "profile_warning_seconds": 0.8, "feature_scales": {"ear": 0.04, "mar": 0.15, "pitch": 12.0, "yaw": 15.0, "roll": 15.0}},
        "gpio": {"enabled": False, "simulation_mode": True, "numbering": "BOARD"},
        "buzzer": {"enabled": True, "type": "pwm_native", "board_pin": 33, "active_high": False, "muted": False, "idle_frequency": 3000, "pwm_duty_cycle": 50, "pwm_chip": 0, "pwm_channel": 2, "min_frequency": 2000, "max_frequency": 5000},
        "leds": {"enabled": True, "active_high": True, "green_board_pin": 29, "yellow_board_pin": 31, "red_board_pin": 32},
        "switch": {"enabled": False, "topology": "on_off_on", "auto_board_pin": 35, "emergency_board_pin": 37, "center_mode": "MAINTENANCE", "active_low": True, "debounce_ms": 100},
        "interface": {"presentation_enabled": True, "show_landmarks": True, "show_information_panel": True, "show_virtual_leds": True, "close_action": "FULL_SHUTDOWN"},
        "logging": {"enabled": False, "events_only": True, "path": "logs/events.csv", "save_images": False, "record_video": False},
    }


def deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
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

        performance = config.get("performance", {})
        self.target_analysis_fps = max(
            1.0,
            float(performance.get("target_analysis_fps", 20.0)),
        )
        self.analysis_period = 1.0 / self.target_analysis_fps
        self.frame_wait_timeout = max(
            0.01,
            float(performance.get("frame_wait_timeout_seconds", 0.1)),
        )
        camera_fps = max(1.0, float(config["camera"].get("fps", 30.0)))
        self.analysis_tolerance = 0.5 / camera_fps
        self.analysis_skipped_frames = 0
        self.capture_dropped_frames = 0

    def start_calibration(self, profile="OPEN"):
        self.forced_test_state = None
        self.detector.reset()
        if self.face is not None:
            self.face.reset_eye_filter()
        self.calibration.start(profile)

    def run(self):
        exit_code = 0
        try:
            performance = self.config.get("performance", {})
            cv2.setUseOptimized(bool(performance.get("opencv_optimized", True)))
            cv2.setNumThreads(max(0, int(performance.get("opencv_threads", 2))))
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
        last_sequence = -1
        next_analysis_at = 0.0
        last_frame = None

        while not self.shutdown.requested:
            self.mode.update_from_switch(self.gpio)
            frame, timestamp, sequence = self.camera.wait_for_frame(
                last_sequence,
                timeout=self.frame_wait_timeout,
                copy=False,
            )
            if frame is None:
                if self.camera.error:
                    raise RuntimeError(self.camera.error)
                if self.ui and last_frame is not None:
                    self._render_if_needed(last_metrics, last_frame)
                continue

            if last_sequence >= 0 and sequence > last_sequence + 1:
                self.capture_dropped_frames += sequence - last_sequence - 1
            last_sequence = sequence
            last_frame = frame

            if self.mode.mode == ModeController.EMERGENCY:
                state, reason = self.detector.update(
                    {"face_detected": False},
                    self.mode.mode,
                )
                self.alerts.update(state, self.mode.mode)
                self._render_if_needed(last_metrics, frame)
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
                self.logger.log_transition(
                    self.mode.mode,
                    self.detector.previous_state,
                    state,
                    reason,
                    metrics,
                    self.analysis_fps,
                    self.gpio,
                )
                self._render_if_needed(metrics, frame)
                if self.camera.error:
                    raise RuntimeError(self.camera.error)
                continue

            now = time.monotonic()
            if now < next_analysis_at - self.analysis_tolerance:
                self.analysis_skipped_frames += 1
                continue
            if next_analysis_at <= 0.0:
                next_analysis_at = now + self.analysis_period
            else:
                next_analysis_at += self.analysis_period
                while next_analysis_at <= now:
                    next_analysis_at += self.analysis_period

            analysis_started = time.monotonic()
            metrics = self.face.analyze(frame)
            metrics["frame_sequence"] = sequence
            metrics["frame_age_ms"] = max(
                0.0,
                (analysis_started - timestamp) * 1000.0,
            )
            metrics["analysis_skipped_frames"] = self.analysis_skipped_frames
            metrics["capture_dropped_frames"] = self.capture_dropped_frames
            metrics["target_analysis_fps"] = self.target_analysis_fps

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
            self.logger.log_transition(
                self.mode.mode,
                self.detector.previous_state,
                state,
                reason,
                metrics,
                self.analysis_fps,
                self.gpio,
            )

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
            frame, _ = self.camera.get_latest_frame(copy=False)
        if frame is not None:
            rendered_metrics = dict(metrics)
            rendered_metrics.update(self.calibration.status_metrics())
            rendered_metrics.setdefault(
                "runtime_seconds",
                time.monotonic() - self.started_at,
            )
            rendered_metrics.setdefault("analysis_ms", self.analysis_ms)
            self.ui.draw(
                frame,
                rendered_metrics,
                self.detector,
                self.mode,
                self.gpio,
                self.camera.capture_fps,
                self.analysis_fps,
                self.hardware_mode(),
            )
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
        return self.detector._set(
            state,
            reasons.get(state, "Prueba manual de salidas"),
            now,
        )

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
