import unittest

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


def make_config():
    return {
        "calibration": {
            "duration_seconds": 0.20,
            "min_samples": 3,
            "quality_threshold": 0.20,
            "min_ear_gap": 0.015,
            "max_ear_std": 0.08,
            "max_profile_distance": 4.0,
            "profile_min_confidence": 0.15,
            "profile_warning_seconds": 0.8,
            "feature_scales": {
                "ear": 0.04,
                "mar": 0.15,
                "pitch": 12.0,
                "yaw": 15.0,
                "roll": 15.0,
            },
        },
        "fatigue": {
            "ear_threshold": 0.22,
            "blink_min_seconds": 0.08,
            "blink_max_seconds": 0.70,
            "prealert_closed_seconds": 0.45,
            "alert_closed_seconds": 0.85,
            "critical_closed_seconds": 1.60,
            "recovery_seconds": 1.0,
            "perclos_window_seconds": 60.0,
            "perclos_warning_threshold": 0.25,
            "perclos_alert_threshold": 0.35,
            "mar_threshold": 0.65,
            "yawn_min_seconds": 1.0,
            "head_nod_pitch_threshold": 18.0,
            "head_nod_min_seconds": 0.8,
            "gaze_away_warning_seconds": 2.0,
            "no_face_warning_seconds": 2.0,
        },
        "interface": {
            "show_landmarks": True,
            "show_information_panel": True,
        },
    }


def metrics(ear, pitch=0.0, mar=0.25, yaw=0.0, roll=0.0, gaze="CENTRO"):
    return {
        "face_detected": True,
        "quality": 0.9,
        "ear": ear,
        "mar": mar,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "gaze": gaze,
    }


class CalibrationManagerTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.manager = CalibrationManager(make_config(), clock=self.clock)

    def capture(self, profile, sample):
        self.manager.start(profile)
        result = None
        for _ in range(4):
            result = self.manager.update(dict(sample))
            self.clock.advance(0.08)
        if result is None:
            result = self.manager.update(dict(sample))
        self.assertIsNotNone(result)
        return result

    def test_three_profiles_build_thresholds_and_classify(self):
        self.capture("OPEN", metrics(0.32, pitch=0.0))
        self.capture("DROWSY", metrics(0.23, pitch=9.0, mar=0.32))
        result = self.capture("ASLEEP", metrics(0.12, pitch=28.0, gaze="ABAJO"))

        self.assertTrue(result["model_ready"])
        self.assertTrue(self.manager.model_ready)
        self.assertAlmostEqual(0.275, self.manager.thresholds["drowsy_ear_threshold"], places=3)
        self.assertAlmostEqual(0.175, self.manager.thresholds["ear_threshold"], places=3)

        self.assertEqual("OPEN", self.manager.classify(metrics(0.32, pitch=0.0))["profile"])
        self.assertEqual("DROWSY", self.manager.classify(metrics(0.23, pitch=9.0, mar=0.32))["profile"])
        self.assertEqual("ASLEEP", self.manager.classify(metrics(0.12, pitch=28.0, gaze="ABAJO"))["profile"])
        self.assertEqual("O:OK S:OK D:OK", self.manager.status_metrics()["calibration_profiles"])

    def test_profiles_with_overlapping_ear_are_rejected(self):
        self.capture("OPEN", metrics(0.30))
        self.capture("DROWSY", metrics(0.29))
        result = self.capture("ASLEEP", metrics(0.28))

        self.assertFalse(result["model_ready"])
        self.assertFalse(self.manager.model_ready)
        self.assertIn("no separables", self.manager.message)

    def test_low_quality_samples_do_not_complete_profile(self):
        self.manager.start("OPEN")
        bad = metrics(0.32)
        bad["quality"] = 0.1
        for _ in range(5):
            result = self.manager.update(bad)
            self.clock.advance(0.08)

        self.assertIsNone(result)
        self.assertNotIn("OPEN", self.manager.profiles)
        self.assertIn("invalida", self.manager.message)


class FatigueDetectorCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.detector = FatigueDetector(make_config())
        self.detector.reset()

    def test_apply_calibration_updates_both_ear_thresholds(self):
        self.detector.apply_calibration({
            "thresholds": {
                "drowsy_ear_threshold": 0.275,
                "ear_threshold": 0.175,
            }
        })

        self.assertAlmostEqual(0.275, self.detector.drowsy_ear_threshold)
        self.assertAlmostEqual(0.175, self.detector.ear_threshold)

    def test_drowsy_and_asleep_profiles_require_sustained_time(self):
        drowsy = {"calibrated_profile": "DROWSY", "calibrated_profile_confidence": 0.9}
        self.detector._update_calibrated_profile(10.0, drowsy)
        self.detector._update_calibrated_profile(10.9, drowsy)
        state, _ = self.detector._decide(10.9, {"perclos": 0.0}, False, True)
        self.assertEqual("POSIBLE_SOMNOLENCIA", state)

        self.detector.reset()
        asleep = {"calibrated_profile": "ASLEEP", "calibrated_profile_confidence": 0.9}
        self.detector._update_calibrated_profile(20.0, asleep)
        self.detector._update_calibrated_profile(20.9, asleep)
        state, _ = self.detector._decide(20.9, {"perclos": 0.0}, False, True)
        self.assertEqual("ALERTA", state)

        self.detector._update_calibrated_profile(21.7, asleep)
        state, _ = self.detector._decide(21.7, {"perclos": 0.0}, False, True)
        self.assertEqual("ALERTA_CRITICA", state)

    def test_face_loss_resets_sustained_calibrated_profile(self):
        self.detector.calibrated_profile = "ASLEEP"
        self.detector.calibrated_profile_confidence = 0.9
        self.detector.calibrated_profile_since = 10.0
        self.detector.calibrated_profile_seconds = 2.0

        self.detector.update({"face_detected": False})

        self.assertEqual("DESCONOCIDO", self.detector.calibrated_profile)
        self.assertEqual(0.0, self.detector.calibrated_profile_seconds)

    def test_recovery_timer_releases_alert_after_configured_delay(self):
        self.detector.state = "ALERTA"

        state, reason = self.detector._decide(10.0, {"perclos": 0.0}, False, True)
        self.assertEqual("ALERTA", state)
        self.assertIn("Periodo de recuperacion", reason)
        self.assertEqual(10.0, self.detector.recovery_since)

        state, _ = self.detector._decide(10.9, {"perclos": 0.0}, False, True)
        self.assertEqual("ALERTA", state)

        state, _ = self.detector._decide(11.01, {"perclos": 0.0}, False, True)
        self.assertEqual("NORMAL", state)
        self.assertIsNone(self.detector.recovery_since)

    def test_risk_reappearance_restarts_recovery_timer(self):
        self.detector.state = "ALERTA"
        self.detector._decide(20.0, {"perclos": 0.0}, False, True)

        self.detector.closed_duration = 0.90
        state, _ = self.detector._decide(20.5, {"perclos": 0.0}, True, True)
        self.assertEqual("ALERTA", state)
        self.assertIsNone(self.detector.recovery_since)

        self.detector.closed_duration = 0.0
        state, _ = self.detector._decide(21.0, {"perclos": 0.0}, False, True)
        self.assertEqual("ALERTA", state)
        self.assertEqual(21.0, self.detector.recovery_since)

        state, _ = self.detector._decide(22.01, {"perclos": 0.0}, False, True)
        self.assertEqual("NORMAL", state)
        self.assertIsNone(self.detector.recovery_since)


class PresentationCalibrationKeyTests(unittest.TestCase):
    def test_o_s_d_keys_select_distinct_profiles(self):
        selected = []

        class FakeApp(object):
            def start_calibration(self, profile):
                selected.append(profile)

        ui = PresentationUI(make_config())
        app = FakeApp()
        ui.handle_key(ord("o"), app)
        ui.handle_key(ord("s"), app)
        ui.handle_key(ord("d"), app)

        self.assertEqual(["OPEN", "DROWSY", "ASLEEP"], selected)


if __name__ == "__main__":
    unittest.main()
