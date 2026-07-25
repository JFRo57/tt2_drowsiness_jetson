import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from src.application import DrowsinessApplication, default_config
from src.calibration import CalibrationManager
from src.fatigue_detector import FatigueDetector
from src.presentation_ui import PresentationUI


class FakeClock(object):
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


def measured(ear, left=None, right=None, quality=0.9, pitch=0.0,
             yaw=0.0, roll=0.0, mar=0.25):
    return {
        "face_detected": True,
        "quality": quality,
        "ear": ear,
        "left_ear": ear if left is None else left,
        "right_ear": ear if right is None else right,
        "eye_reliable": True,
        "left_eye_reliable": True,
        "right_eye_reliable": True,
        "pose_reliable": True,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "mar": mar,
        "brightness": 100.0,
    }


class CalibrationManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.clock = FakeClock()
        self.config = default_config()
        calibration = self.config["calibration"]
        calibration.update({
            "profile_path": os.path.join(self.temp.name, "profile.json"),
            "parameters_path": os.path.join(self.temp.name, "parameters.json"),
            "duration_seconds": 0.20,
            "preparation_seconds": 0.0,
            "require_stage_confirmation": False,
            "min_samples": 3,
            "min_valid_sample_ratio": 0.60,
            "min_ear_gap": 0.015,
            "min_open_closed_gap": 0.05,
            "natural_blink_observation_seconds": 0.80,
            "voluntary_blink_observation_seconds": 1.5,
            "min_natural_blinks": 2,
            "min_voluntary_blinks": 5,
        })
        self.manager = CalibrationManager(self.config, clock=self.clock)

    def tearDown(self):
        self.temp.cleanup()

    def capture_stage(self, name, sample):
        self.manager.start(name)
        result = None
        for _ in range(5):
            result = self.manager.update(dict(sample)) or result
            self.clock.advance(0.06)
        self.assertIsNotNone(result)
        return result

    def capture_static(self, opened=0.32, reduced=0.22, closed=0.10):
        self.capture_stage("OPEN", measured(opened, pitch=2.0))
        self.capture_stage("DROWSY", measured(reduced, pitch=2.0))
        return self.capture_stage("ASLEEP", measured(closed, pitch=2.0))

    def dynamic_blink(self):
        for ear, step in ((0.32, 0.05), (0.22, 0.05), (0.10, 0.10),
                          (0.22, 0.05), (0.32, 0.10)):
            result = self.manager.update(measured(ear))
            self.clock.advance(step)
            if result and result.get("model_ready"):
                return result
        return None

    def complete_dynamic(self):
        result = self.dynamic_blink()
        result = self.dynamic_blink() or result
        for _ in range(12):
            result = self.manager.update(measured(0.32)) or result
            self.clock.advance(0.08)
            if result and result.get("model_ready"):
                break
        return result

    def test_three_static_stages_are_per_eye_and_use_reduced_name(self):
        result = self.capture_static()

        self.assertTrue(result["static_ready"])
        self.assertTrue(self.manager.static_ready)
        self.assertTrue(self.manager.active)
        self.assertEqual("DYNAMIC_NATURAL", self.manager.active_profile)
        self.assertEqual("O:OK R:OK C:OK",
                         self.manager.status_metrics()["calibration_profiles"])
        self.assertAlmostEqual(0.27, self.manager.thresholds["reduced_ear_threshold"])
        self.assertAlmostEqual(0.16, self.manager.thresholds["ear_threshold"])

    def test_complete_profile_saves_blink_percentiles_atomically(self):
        self.capture_static()
        result = self.complete_dynamic()

        self.assertIsNotNone(result)
        self.assertTrue(result["model_ready"])
        self.assertTrue(result["profile_saved"])
        self.assertTrue(result["parameters_saved"])
        self.assertTrue(self.manager.ready_for_monitoring())
        selected = result["profile_data"]["blink_baseline"]["selected"]
        self.assertEqual(2, selected["event_count"])
        self.assertIsNotNone(selected["median_seconds"])
        self.assertIsNotNone(selected["p75_seconds"])
        self.assertIsNotNone(selected["p90_seconds"])
        self.assertFalse(os.path.exists(self.manager.profile_path + ".tmp"))
        self.assertFalse(os.path.exists(self.manager.parameters_path + ".tmp"))
        with open(self.manager.parameters_path, "r") as handle:
            parameters = json.load(handle)
        self.assertEqual("tt2_calibration_parameters", parameters["kind"])
        self.assertEqual(2, parameters["format_version"])
        self.assertIn("eyes", parameters)
        self.assertIn("thresholds", parameters)
        self.assertIn("profile_data", parameters)
        self.assertEqual(
            self.config["calibration"]["natural_blink_observation_seconds"],
            parameters["calibration_settings"]["natural_blink_observation_seconds"],
        )

    def test_parameter_export_recovers_profile_when_main_file_is_missing(self):
        self.capture_static()
        self.complete_dynamic()
        os.unlink(self.manager.profile_path)

        reloaded = CalibrationManager(self.config, clock=self.clock)

        self.assertTrue(reloaded.ready_for_monitoring())
        self.assertEqual("Parametros de calibracion cargados", reloaded.message)
        self.assertAlmostEqual(0.16, reloaded.thresholds["ear_threshold"])

    def test_application_reuses_persistent_calibration_without_repeating(self):
        self.capture_static()
        self.complete_dynamic()

        app = DrowsinessApplication(self.config, simulation=True)

        self.assertTrue(app.calibration.ready_for_monitoring())
        self.assertIsNotNone(app.detector.profile)
        self.assertAlmostEqual(0.16, app.detector.ear_threshold)

    def test_normalized_closure_is_personal_and_clamped(self):
        self.capture_static()
        result = self.complete_dynamic()
        profile = result["profile_data"]
        self.assertAlmostEqual(
            0.0, self.manager.normalized_closure(measured(profile["eyes"]["combined"]["open"]))
        )
        self.assertAlmostEqual(
            1.0, self.manager.normalized_closure(measured(profile["eyes"]["combined"]["closed"]))
        )
        self.assertEqual(0.0, self.manager.normalized_closure(measured(0.60)))
        self.assertEqual(1.0, self.manager.normalized_closure(measured(0.03)))

    def test_invalid_stage_reports_only_affected_stage(self):
        self.manager.start("OPEN")
        result = None
        for _ in range(5):
            result = self.manager.update(measured(0.32, quality=0.1)) or result
            self.clock.advance(0.06)

        self.assertFalse(result["accepted"])
        self.assertEqual("OPEN", result["failed_stage"])
        self.assertIn("Repita solo esta etapa", result["reason"])
        self.assertNotIn("OPEN", self.manager.profiles)

    def test_values_too_close_are_rejected_before_dynamic_phase(self):
        self.capture_stage("OPEN", measured(0.30))
        self.capture_stage("REDUCED", measured(0.29))
        result = self.capture_stage("CLOSED", measured(0.28))

        self.assertFalse(result["accepted"])
        self.assertFalse(self.manager.static_ready)
        self.assertFalse(self.manager.active)
        self.assertIn("ABIERTO > REDUCIDO > CERRADO", self.manager.message)
        self.assertNotIn(result["failed_stage"], self.manager.profiles)
        self.assertEqual(result["failed_stage"], self.manager.next_required_stage())

    def test_preparation_delay_does_not_sample_before_stage_capture(self):
        self.manager.config["preparation_seconds"] = 0.30
        self.manager.preparation_seconds = 0.30
        self.manager.start("OPEN")

        for _ in range(3):
            self.assertIsNone(self.manager.update(measured(0.32)))
            self.clock.advance(0.08)

        status = self.manager.status_metrics()
        self.assertTrue(status["calibration_preparing"])
        self.assertEqual(0, status["calibration_valid_samples"])
        self.assertEqual(0, status["calibration_sample_attempts"])

        self.clock.advance(0.10)
        self.assertIsNone(self.manager.update(measured(0.32)))
        status = self.manager.status_metrics()
        self.assertFalse(status["calibration_preparing"])
        self.assertEqual(1, status["calibration_valid_samples"])

    def test_excessive_head_motion_rejects_stage(self):
        self.manager.start("OPEN")
        result = None
        for pitch in (0.0, 15.0, -15.0, 16.0, -16.0):
            result = self.manager.update(measured(0.32, pitch=pitch)) or result
            self.clock.advance(0.06)

        self.assertFalse(result["accepted"])
        self.assertIn("movimiento excesivo", result["reason"])

    def test_failed_recalibration_does_not_overwrite_valid_profile(self):
        self.capture_static()
        self.complete_dynamic()
        with open(self.manager.profile_path, "r") as handle:
            previous = handle.read()

        self.manager.start("REDUCED")
        for _ in range(5):
            self.manager.update(measured(0.22, quality=0.05))
            self.clock.advance(0.06)
        with open(self.manager.profile_path, "r") as handle:
            current = handle.read()

        self.assertEqual(previous, current)
        self.assertFalse(self.manager.ready_for_monitoring())

    def test_natural_shortage_requests_voluntary_blinks(self):
        self.capture_static()
        result = None
        for _ in range(12):
            result = self.manager.update(measured(0.32)) or result
            self.clock.advance(0.08)

        self.assertTrue(result["dynamic_fallback"])
        self.assertEqual("DYNAMIC_VOLUNTARY", self.manager.active_profile)
        self.assertIn("cinco y ocho", self.manager.message)

    def test_legacy_v1_profile_is_migrated_when_it_has_blink_baseline(self):
        stats = lambda value: {"median": value, "robust_std": 0.01}
        legacy = {
            "format_version": 1,
            "profiles": {
                "OPEN": {"ear": stats(0.32)},
                "DROWSY": {"ear": stats(0.22)},
                "ASLEEP": {"ear": stats(0.10)},
            },
            "blink_baseline": {"selected": {
                "source": "legacy", "event_count": 5,
                "median_seconds": 0.18, "p90_seconds": 0.25,
            }},
        }
        with open(self.manager.profile_path, "w") as handle:
            json.dump(legacy, handle)

        migrated = CalibrationManager(self.config, clock=self.clock)

        self.assertTrue(migrated.ready_for_monitoring())
        self.assertEqual(2, migrated.profile_data["format_version"])
        self.assertEqual(1, migrated.profile_data["migrated_from_version"])

        migrated.start("REDUCED")
        result = None
        for _ in range(5):
            result = migrated.update(measured(0.22)) or result
            self.clock.advance(0.06)
        self.assertTrue(result["static_ready"])
        self.assertEqual("DYNAMIC_NATURAL", migrated.active_profile)


class DetectorProfileTests(unittest.TestCase):
    def test_apply_calibration_keeps_legacy_threshold_properties(self):
        detector = FatigueDetector(default_config(), clock=FakeClock())
        profile = {
            "format_version": 2,
            "eyes": {
                "left": {"open": 0.32, "reduced": 0.22, "closed": 0.10},
                "right": {"open": 0.31, "reduced": 0.21, "closed": 0.09},
                "combined": {"open": 0.315, "reduced": 0.215, "closed": 0.095},
            },
            "partial_closure_normalized": 0.45,
            "blink_baseline": {"selected": {"event_count": 5,
                "median_seconds": 0.18, "p90_seconds": 0.25}},
            "neutral_head_pose": {"pitch": 3.0, "yaw": 0.0, "roll": 0.0},
            "thresholds": {"drowsy_ear_threshold": 0.27,
                "ear_threshold": 0.16, "ear_open_threshold": 0.18,
                "open_pitch_reference": 3.0},
        }

        self.assertTrue(detector.apply_calibration(profile))
        self.assertAlmostEqual(0.27, detector.drowsy_ear_threshold)
        self.assertAlmostEqual(0.16, detector.ear_threshold)
        self.assertAlmostEqual(0.18, detector.ear_open_threshold)
        self.assertAlmostEqual(3.0, detector.head_pitch_baseline)


class PresentationCalibrationKeyTests(unittest.TestCase):
    def test_keys_use_reduced_closed_dynamic_and_full_names(self):
        selected = []
        confirmed = []

        class FakeApp(object):
            def start_calibration(self, profile):
                selected.append(profile)

            def confirm_calibration_step(self):
                confirmed.append(True)
                return True

        ui = PresentationUI(default_config())
        app = FakeApp()
        for key in ("o", "s", "d", "b", "c"):
            ui.handle_key(ord(key), app)
        for key in (32, 10, 13):
            ui.handle_key(key, app)

        self.assertEqual(
            ["OPEN", "REDUCED", "CLOSED", "DYNAMIC", "FULL"], selected
        )
        self.assertEqual(3, len(confirmed))


class ApplicationCalibrationFlowTests(unittest.TestCase):
    def test_full_sequence_waits_for_confirmation_then_starts_reduced(self):
        temp = tempfile.TemporaryDirectory()
        try:
            clock = FakeClock()
            config = default_config()
            config["calibration"].update({
                "profile_path": os.path.join(temp.name, "profile.json"),
                "duration_seconds": 0.20,
                "preparation_seconds": 0.0,
                "min_samples": 3,
                "min_valid_sample_ratio": 0.60,
            })
            app = DrowsinessApplication(config, simulation=True)
            app.calibration.clock = clock

            with redirect_stdout(io.StringIO()):
                app.start_calibration("FULL")
                result = None
                for _ in range(5):
                    result = app.calibration.update(measured(0.32)) or result
                    clock.advance(0.06)

                app._handle_calibration_result(result)

            self.assertFalse(app.calibration.active)
            self.assertTrue(app.calibration.awaiting_confirmation)
            self.assertEqual("REDUCED", app.calibration.pending_confirmation_profile)
            self.assertTrue(
                app.calibration.status_metrics()["calibration_waiting_confirmation"]
            )
            with redirect_stdout(io.StringIO()):
                self.assertTrue(app.confirm_calibration_step())

            self.assertTrue(app.calibration.active)
            self.assertEqual("REDUCED", app.calibration.active_profile)
            self.assertEqual(
                "O:OK R:-- C:--",
                app.calibration.status_metrics()["calibration_profiles"],
            )
        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
