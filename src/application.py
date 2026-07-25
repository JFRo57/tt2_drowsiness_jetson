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
    config = {
        "camera": {"sensor_id": 0, "capture_width": 1280, "capture_height": 720, "processing_width": 640, "processing_height": 360, "display_width": 1100, "display_height": 620, "fps": 30, "flip_method": 2, "output_format": "BGRx", "reconnect_attempts": 3},
        "dlib": {"predictor_path": "models/shape_predictor_68_face_landmarks.dat", "upsample": 0, "detection_interval_frames": 12, "no_face_detection_interval_frames": 3, "detector_scale": 0.5, "tracking_mode": "landmarks", "use_correlation_tracker": False, "tracker_quality_threshold": 6.5, "pose_interval_frames": 2, "gaze_interval_frames": 2},
        "face_detection": {"backend": "auto", "backend_order": ["dlib_cnn_cuda", "opencv_cuda_fp16", "dlib_hog"], "allow_fallback": True, "warmup": True, "confidence_threshold": 0.55, "dlib_cnn_model_path": "models/mmod_human_face_detector.dat", "opencv_config_path": "models/deploy.prototxt", "opencv_model_path": "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"},
        "preprocessing": {"use_clahe": True, "adaptive_clahe": True, "clahe_dark_threshold": 75.0, "clahe_bright_threshold": 205.0, "clahe_clip_limit": 2.0, "clahe_grid_size": 8, "use_gamma": False, "gamma": 1.0},
        "performance": {"target_analysis_fps": 20.0, "frame_wait_timeout_seconds": 0.1, "opencv_threads": 2, "opencv_optimized": True},
        "stability": {"landmark_shape_alpha": 0.30, "landmark_translation_alpha": 0.75, "tracking_rect_alpha": 0.35, "redetection_rect_alpha": 0.30, "redetection_min_iou": 0.15, "redetection_max_center_shift": 0.45, "redetection_miss_tolerance": 2, "ear_median_window": 3, "ear_hysteresis": 0.012, "close_confirm_seconds": 0.08, "open_confirm_seconds": 0.15, "unreliable_hold_seconds": 0.25, "face_loss_hold_seconds": 0.25, "eye_quality_threshold": 0.25, "min_eye_width_pixels": 10.0, "min_eye_sharpness": 12.0, "max_eye_ear_difference": 0.12, "max_eye_ear_difference_ratio": 0.65, "max_eye_yaw_degrees": 32.0},
        "fatigue": {"ear_threshold": 0.22, "use_session_calibration": True, "blink_min_seconds": 0.08, "blink_max_seconds": 0.70, "prealert_closed_seconds": 0.45, "alert_closed_seconds": 0.85, "critical_closed_seconds": 1.6, "recovery_seconds": 1.0, "perclos_window_seconds": 60, "perclos_warning_threshold": 0.25, "perclos_alert_threshold": 0.35, "mar_threshold": 0.65, "yawn_min_seconds": 1.0, "head_nod_pitch_threshold": 18.0, "head_nod_min_seconds": 0.8, "gaze_away_warning_seconds": 2.0, "no_face_warning_seconds": 2.0},
        "calibration": {"duration_seconds": 5.0, "min_samples": 30, "quality_threshold": 0.30, "min_ear_gap": 0.020, "max_ear_std": 0.035, "max_ear_mad": 0.025, "max_profile_overlap_ratio": 0.65, "max_profile_distance": 4.0, "profile_min_confidence": 0.35, "profile_warning_seconds": 0.8, "feature_scales": {"ear": 0.04, "mar": 0.15, "pitch": 12.0, "yaw": 15.0, "roll": 15.0}},
        "gpio": {"enabled": False, "simulation_mode": True, "numbering": "BOARD", "output_refresh_seconds": 0.5},
        "buzzer": {"enabled": True, "type": "pwm_native", "board_pin": 33, "active_high": False, "muted": False, "idle_frequency": 3000, "pwm_duty_cycle": 50, "pwm_chip": 0, "pwm_channel": 2, "min_frequency": 2000, "max_frequency": 5000},
        "leds": {"enabled": True, "active_high": True, "green_board_pin": 29, "yellow_board_pin": 31, "red_board_pin": 32},
        "switch": {"enabled": False, "topology": "on_off_on", "auto_board_pin": 35, "emergency_board_pin": 37, "center_mode": "MAINTENANCE", "active_low": True, "debounce_ms": 100},
        "interface": {"presentation_enabled": True, "show_landmarks": True, "show_information_panel": True, "show_virtual_leds": True, "close_action": "FULL_SHUTDOWN"},
        "logging": {"enabled": False, "events_only": True, "path": "logs/events.csv", "save_images": False, "record_video": False},
    }
    config["stability"].update({"max_single_eye_yaw_degrees": 42.0})
    config["fatigue"].update({
        "critical_closed_seconds": 2.0,
        "allow_single_eye": True, "single_eye_critical_enabled": False,
        "blink_start_closure_level": 0.35,
        "blink_closed_closure_level": 0.75, "blink_reopen_level": 0.25,
        "blink_event_max_seconds": 4.0, "deep_closure_level": 0.80,
        "partial_closure_hysteresis": 0.05,
        "reduced_opening_sustain_seconds": 1.5,
        "sustained_closure_seconds": 0.8, "severe_closure_seconds": 1.2,
        "closure_with_head_drop_seconds": 0.8,
        "unreliable_event_abort_seconds": 0.4,
        "prolonged_blink_absolute_seconds": 0.55,
        "prolonged_blink_relative_multiplier": 1.8,
        "prolonged_blinks_strong_count": 2,
        "fast_window_seconds": 3.0, "medium_window_seconds": 25.0,
        "long_window_seconds": 60.0, "perclos_min_coverage": 0.65,
        "perclos_min_valid_seconds": 15.0,
        "mouth_open_threshold": 0.48, "mouth_wide_threshold": 0.65,
        "mouth_closed_threshold": 0.38, "yawn_wide_sustain_seconds": 0.65,
        "repeated_yawn_count": 2, "head_down_direction": 1.0,
        "head_down_pitch_threshold": 18.0,
        "head_recover_pitch_threshold": 8.0,
        "head_lateral_turn_threshold": 28.0,
        "head_drop_velocity_degrees_per_second": 28.0,
        "head_down_sustain_seconds": 0.8, "repeated_nod_count": 2,
        "severe_closures_critical_count": 2,
        "critical_minimum_hold_seconds": 2.0,
        "critical_open_recovery_seconds": 2.0,
        "recovery_open_closure_level": 0.25,
        "recovery_observation_seconds": 8.0,
        "recovery_relapse_closed_seconds": 0.8,
        "suspicion_to_somnolence_seconds": 4.0,
        "suspicion_clear_seconds": 8.0, "somnolence_clear_seconds": 10.0,
    })
    config["vision_reliability"] = {
        "initial_valid_seconds": 0.5, "min_quality": 0.25,
        "min_brightness": 12.0, "max_brightness": 250.0,
        "max_pose_pitch_degrees": 42.0, "max_pose_roll_degrees": 35.0,
        "no_face_seconds": 2.0, "persistent_loss_seconds": 8.0,
        "camera_failure_seconds": 6.0, "min_analysis_fps": 8.0,
        "low_fps_grace_seconds": 3.0, "startup_fps_grace_seconds": 3.0,
    }
    config["calibration"].update({
        "profile_path": "calibration_profile.json", "auto_start_if_missing": True,
        "parameters_path": "calibration_parameters.json",
        "preparation_seconds": 2.0, "require_stage_confirmation": True,
        "min_valid_sample_ratio": 0.65, "min_open_closed_gap": 0.06,
        "max_head_angle_std_degrees": 6.0,
        "max_eye_asymmetry_ratio": 0.35, "max_partial_eye_difference": 0.25,
        "min_possible_ear": 0.03, "max_possible_ear": 0.60,
        "natural_blink_observation_seconds": 60.0,
        "voluntary_blink_observation_seconds": 60.0,
        "min_natural_blinks": 3, "min_voluntary_blinks": 5,
        "calibration_blink_max_seconds": 1.2, "blink_min_seconds": 0.08,
        "blink_start_closure_level": 0.35,
        "blink_closed_closure_level": 0.75, "blink_reopen_level": 0.25,
        "max_dynamic_head_delta_degrees": 12.0,
    })
    config["alerts"] = {
        "suspicion_frequency": 2500, "somnolence_frequency": 3500,
        "critical_frequency": 4500, "supervision_frequency": 3000,
        "initial_led_hz": 3.0, "calibration_led_hz": 1.2,
        "degraded_vision_led_hz": 1.2, "suspicion_led_hz": 1.8,
        "somnolence_led_hz": 3.0, "critical_led_hz": 7.0,
        "recovery_led_hz": 1.0, "supervision_led_hz": 3.0,
        "failure_led_hz": 3.0, "error_led_cycle_seconds": 1.2,
        "error_led_pulse_seconds": 0.15, "error_led_gap_seconds": 0.15,
        "suspicion_notice_seconds": 3.0, "suspicion_cycle_seconds": 1.2,
        "suspicion_on_seconds": 0.2, "somnolence_cycle_seconds": 2.4,
        "somnolence_group_seconds": 1.5, "somnolence_beat_seconds": 0.5,
        "somnolence_on_seconds": 0.2, "critical_cycle_seconds": 2.1,
        "critical_group_seconds": 1.5, "critical_beat_seconds": 0.3,
        "critical_on_seconds": 0.15, "supervision_cycle_seconds": 2.0,
        "supervision_on_seconds": 0.25,
    }
    config["interface"]["debug_overlay"] = True
    config["logging"].update({
        "snapshot_interval_seconds": 5.0, "flush_interval_seconds": 2.0,
    })
    return config


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
        self.alerts = AlertController(self.gpio, config)
        self.detector = FatigueDetector(config)
        self.calibration = CalibrationManager(config)
        if self.calibration.profile_data:
            self.detector.apply_calibration(self.calibration.profile_data)
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
        self.detector.reset(clear_history=True)
        if self.face is not None:
            self.face.reset_eye_filter()
        self.calibration.start(profile)
        print("CALIBRACION:", self.calibration.message)

    def confirm_calibration_step(self):
        if not self.calibration.confirm_pending_stage():
            return False
        if self.face is not None:
            self.face.reset_eye_filter()
        print("CALIBRACION:", self.calibration.message)
        return True

    def run(self):
        exit_code = 0
        try:
            performance = self.config.get("performance", {})
            cv2.setUseOptimized(bool(performance.get("opencv_optimized", True)))
            cv2.setNumThreads(max(0, int(performance.get("opencv_threads", 2))))
            self._setup_outputs()
            self.logger.open()
            self.face = FaceAnalyzer(self.config)
            if not self.camera.start():
                raise RuntimeError(self.camera.error or "No se pudo iniciar la camara")
            if self.ui:
                self.ui.create()
            if (not self.calibration.ready_for_monitoring()
                    and self.config.get("calibration", {}).get(
                        "auto_start_if_missing", True)):
                self.start_calibration("FULL")
            self._loop()
        except Exception as exc:
            exit_code = 1
            self.error = str(exc)
            print("ERROR:", exc)
            traceback.print_exc()
            self.detector.state = "ERROR"
            self.alerts.update("ERROR", self.mode.mode, self._vision_state())
        finally:
            self.cleanup()
        return exit_code

    def _setup_outputs(self):
        setup_ok = self.gpio.setup()
        gpio_cfg = self.config.get("gpio", {})
        physical_requested = (
            not self.simulation
            and bool(gpio_cfg.get("enabled", False))
            and not bool(gpio_cfg.get("simulation_mode", True))
        )
        if physical_requested and (
            not setup_ok or self.gpio.simulation_mode or self.gpio.GPIO is None
        ):
            raise RuntimeError(
                self.gpio.error
                or "Se solicito GPIO fisico, pero no pudo inicializarse"
            )

    def _loop(self):
        frames = 0
        last_fps = time.monotonic()
        last_metrics = {"face_detected": False, "quality": 0.0}
        last_sequence = -1
        next_analysis_at = 0.0
        last_frame = None

        while not self.shutdown.requested:
            self.mode.update_from_switch(self.gpio)
            # El estado del detector es la fuente de verdad para las salidas.
            # Mantener los patrones activos aunque la camara tarde o pierda cuadros.
            self.alerts.update(
                self.detector.state, self.mode.mode, self._vision_state()
            )
            frame, timestamp, sequence = self.camera.wait_for_frame(
                last_sequence,
                timeout=self.frame_wait_timeout,
                copy=False,
            )
            if self.shutdown.requested:
                break
            if frame is None:
                missing_metrics = {
                    "frame_available": False,
                    "face_detected": False,
                    "quality": 0.0,
                }
                state, reason = self.detector.update(
                    missing_metrics,
                    self.mode.mode,
                    monitoring_enabled=self.calibration.ready_for_monitoring(),
                    analysis_fps=self.analysis_fps,
                    camera_error=self.camera.error,
                )
                if self.calibration.active:
                    self._handle_calibration_result(
                        self.calibration.update(missing_metrics)
                    )
                last_metrics.update(missing_metrics)
                self.alerts.update(
                    state, self.mode.mode, self._vision_state()
                )
                missing_metrics["active_alert"] = self.alerts.active_alert
                self.logger.log_transition(
                    self.mode.mode,
                    self.detector.previous_state,
                    state,
                    reason,
                    missing_metrics,
                    self.analysis_fps,
                    self.gpio,
                )
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
                self.alerts.update(
                    state, self.mode.mode, self._vision_state()
                )
                self._render_if_needed(last_metrics, frame)
                continue

            if self.ui and self.ui.paused:
                paused_metrics = dict(last_metrics)
                state, reason = self.detector.update(
                    paused_metrics,
                    self.mode.mode,
                    monitoring_enabled=self.calibration.ready_for_monitoring(),
                    paused=True,
                    analysis_fps=self.analysis_fps,
                )
                self.alerts.update(
                    state, self.mode.mode, self._vision_state()
                )
                self._render_if_needed(last_metrics, frame)
                continue

            if self.forced_test_state:
                metrics = dict(last_metrics)
                metrics["runtime_seconds"] = time.monotonic() - self.started_at
                metrics["analysis_ms"] = self.analysis_ms
                state, reason = self._apply_forced_test_state()
                self.alerts.update(
                    state, self.mode.mode, self._vision_state()
                )
                metrics["active_alert"] = self.alerts.active_alert
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
                self._handle_calibration_result(calibration_result)
                metrics.update(self.calibration.status_metrics())
            if was_calibrating and not self.calibration.active:
                self.detector.events.reset(clear_history=True)

            state, reason = self.detector.update(
                metrics,
                self.mode.mode,
                monitoring_enabled=self.calibration.ready_for_monitoring(),
                analysis_fps=self.analysis_fps,
            )
            if state == "CALIBRACION":
                self.detector.reason = self.calibration.message
                reason = self.detector.reason

            self.analysis_ms = (time.monotonic() - analysis_started) * 1000.0
            metrics["runtime_seconds"] = time.monotonic() - self.started_at
            metrics["analysis_ms"] = self.analysis_ms
            self.alerts.update(
                state, self.mode.mode, self._vision_state()
            )
            metrics["active_alert"] = self.alerts.active_alert
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

    def _handle_calibration_result(self, result):
        if result is None:
            return
        if result.get("reason"):
            print("CALIBRACION:", result["reason"])
        if result.get("model_ready"):
            self.detector.apply_calibration(result)
            return
        if not result.get("accepted"):
            return
        next_stage = result.get("next_stage") or self.calibration.next_required_stage()
        if next_stage is not None and not self.calibration.active:
            if self.config.get("calibration", {}).get(
                "require_stage_confirmation", False,
            ):
                self.calibration.await_next_stage(next_stage)
                print("CALIBRACION:", self.calibration.message)
            else:
                if self.face is not None:
                    self.face.reset_eye_filter()
                self.calibration.start(next_stage)
                print("CALIBRACION:", self.calibration.message)

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
        if self.ui.consume_click():
            self.confirm_calibration_step()
        if self.ui.is_closed():
            self.shutdown.request("Ventana cerrada")

    def hardware_mode(self):
        return "GPIO simulado" if self.gpio.simulation_mode else "GPIO fisico"

    def _vision_state(self):
        return getattr(getattr(self.detector, "vision", None), "state", None)

    def set_forced_test_state(self, state):
        if state is None:
            self.forced_test_state = None
            self.detector.reset()
            return
        canonical = self.detector.canonical_state(state)
        if canonical in self.detector.STATES:
            self.forced_test_state = canonical

    def _apply_forced_test_state(self):
        now = time.monotonic()
        self.detector.previous_state = self.detector.state
        state = self.forced_test_state
        reasons = {
            "ALERTA": "Prueba manual de salidas: ALERTA",
            "SOSPECHA": "Prueba manual de salidas: SOSPECHA",
            "SOMNOLENCIA": "Prueba manual de salidas: SOMNOLENCIA",
            "CRITICO": "Prueba manual de salidas: CRITICO",
            "RECUPERACION": "Prueba manual de salidas: RECUPERACION",
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
