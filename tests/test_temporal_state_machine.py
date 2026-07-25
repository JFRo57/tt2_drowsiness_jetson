import unittest

from src.application import default_config
from src.fatigue_detector import FatigueDetector


class FakeClock(object):
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


def driver_metrics(ear=0.32, mar=0.25, pitch=0.0, yaw=0.0, roll=0.0,
                   face=True, left_reliable=True, right_reliable=True,
                   quality=0.9, brightness=100.0, sequence=None):
    sample = {
        "frame_available": True,
        "face_detected": face,
        "quality": quality if face else 0.0,
        "brightness": brightness,
        "pose_reliable": bool(face and abs(yaw) <= 42.0),
        "ear": ear if face else None,
        "left_ear": ear if face else None,
        "right_ear": ear if face else None,
        "eye_reliable": bool(left_reliable and right_reliable and face),
        "left_eye_reliable": bool(left_reliable and face),
        "right_eye_reliable": bool(right_reliable and face),
        "mar": mar if face else None,
        "pitch": pitch if face else None,
        "yaw": yaw if face else None,
        "roll": roll if face else None,
        "gaze": "CENTRO" if face else "DESCONOCIDA",
    }
    if sequence is not None:
        sample["frame_sequence"] = sequence
    return sample


def personal_profile():
    return {
        "format_version": 2,
        "eyes": {
            "left": {"open": 0.32, "reduced": 0.22, "closed": 0.10},
            "right": {"open": 0.32, "reduced": 0.22, "closed": 0.10},
            "combined": {"open": 0.32, "reduced": 0.22, "closed": 0.10},
        },
        "partial_closure_normalized": (0.32 - 0.22) / (0.32 - 0.10),
        "blink_baseline": {
            "natural": {"source": "natural", "event_count": 5,
                        "median_seconds": 0.18, "p75_seconds": 0.22,
                        "p90_seconds": 0.25, "mad_seconds": 0.02},
            "voluntary": {"source": "voluntary", "event_count": 0,
                          "median_seconds": None, "p90_seconds": None},
            "selected": {"source": "natural", "event_count": 5,
                         "median_seconds": 0.18, "p75_seconds": 0.22,
                         "p90_seconds": 0.25, "mad_seconds": 0.02},
        },
        "neutral_head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        "thresholds": {"reduced_ear_threshold": 0.27,
                       "drowsy_ear_threshold": 0.27,
                       "ear_threshold": 0.16,
                       "ear_open_threshold": 0.18,
                       "open_pitch_reference": 0.0},
    }


class StateMachineScenarioTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.config = default_config()
        self.config["vision_reliability"]["initial_valid_seconds"] = 0.0
        self.detector = FatigueDetector(self.config, clock=self.clock)
        self.detector.apply_calibration(personal_profile())
        self.last_metrics = None
        self.feed()

    def feed(self, seconds=0.1, fps=20.0, paused=False, **changes):
        sample = driver_metrics(**changes)
        state, reason = self.detector.update(
            sample, analysis_fps=fps, paused=paused
        )
        self.last_metrics = sample
        self.clock.advance(seconds)
        return state, reason

    def feed_for(self, duration, step=0.1, **changes):
        state, reason = self.detector.state, self.detector.reason
        count = int(float(duration) / step) + 1
        for _ in range(count):
            state, reason = self.feed(seconds=step, **changes)
        return state, reason

    def blink(self, duration=0.25):
        self.feed(seconds=0.05, ear=0.32)
        self.feed(seconds=0.05, ear=0.22)
        remaining = max(0.10, float(duration) - 0.15)
        for _ in range(max(1, int(remaining / 0.05))):
            self.feed(seconds=0.05, ear=0.10)
        self.feed(seconds=0.05, ear=0.22)
        return self.feed(seconds=0.05, ear=0.32)

    def yawn(self):
        self.feed(seconds=0.2, mar=0.25)
        self.feed(seconds=0.2, mar=0.52)
        for _ in range(5):
            self.feed(seconds=0.25, mar=0.75)
        self.feed(seconds=0.2, mar=0.50)
        return self.feed(seconds=0.2, mar=0.25)

    def reach_critical(self):
        return self.feed_for(2.4, ear=0.10)

    def test_01_awake_for_several_minutes_stays_alert(self):
        state, reason = self.feed_for(180.0, step=1.0, ear=0.32)
        self.assertEqual("ALERTA", state)
        self.assertIn("comportamiento personal", reason)

    def test_02_one_isolated_yawn_is_weak_and_stays_alert(self):
        state, reason = self.yawn()
        self.assertEqual(1, self.last_metrics["recent_yawns"])
        self.assertEqual("ALERTA", state)
        self.assertIn("comportamiento personal", reason)

    def test_03_repeated_yawns_without_ocular_change_stay_alert(self):
        self.yawn()
        state, reason = self.yawn()
        self.assertGreaterEqual(self.last_metrics["recent_yawns"], 2)
        self.assertEqual("ALERTA", state)
        self.assertTrue(reason)

    def test_04_normal_blinks_are_single_complete_cycles(self):
        for _ in range(5):
            state, reason = self.blink(0.25)
        self.assertEqual(5, self.last_metrics["blink_count_recent"])
        self.assertEqual(0, self.last_metrics["prolonged_blinks_recent"])
        self.assertEqual("ALERTA", state)
        self.assertTrue(reason)

    def test_05_progressively_longer_blinks_enter_suspicion(self):
        self.blink(0.30)
        self.blink(0.65)
        state, reason = self.blink(0.80)
        self.assertEqual("SOSPECHA", state)
        self.assertIn("evidencia fuerte", reason.lower())

    def test_06_sustained_reduced_opening_enters_suspicion(self):
        state, reason = self.feed_for(1.8, ear=0.22)
        self.assertEqual("SOSPECHA", state)
        self.assertIn("apertura ocular reducida", reason.lower())

    def test_07_one_second_closure_is_not_critical(self):
        state, reason = self.feed_for(1.05, ear=0.10)
        self.assertEqual("SOSPECHA", state)
        self.assertNotEqual("CRITICO", state)
        self.assertTrue(reason)

    def test_08_prolonged_critical_closure_is_immediate(self):
        state, reason = self.reach_critical()
        self.assertEqual("CRITICO", state)
        self.assertIn("Posible perdida momentanea", reason)

    def test_09_eye_closure_plus_head_drop_is_critical(self):
        state, reason = self.feed_for(1.1, ear=0.10, pitch=25.0)
        self.assertEqual("CRITICO", state)
        self.assertIn("caida cabeza", reason.lower())

    def test_10_short_face_loss_preserves_previous_state(self):
        self.feed_for(1.8, ear=0.22)
        previous = self.detector.state
        state, reason = self.feed_for(0.8, face=False)
        self.assertEqual(previous, state)
        self.assertEqual("VISION_DEGRADADA", self.last_metrics["vision_state"])
        self.assertIn("conservado", reason)

    def test_11_persistent_visual_loss_is_camera_or_obstruction_state(self):
        state, reason = self.feed_for(8.4, face=False)
        self.assertEqual("CAMARA_OBSTRUIDA_O_FALLO",
                         self.last_metrics["vision_state"])
        self.assertEqual("ALERTA", state)
        self.assertIn("conservado", reason)

    def test_12_critical_recovery_descends_in_required_order(self):
        self.config["fatigue"].update({
            "critical_minimum_hold_seconds": 0.4,
            "critical_open_recovery_seconds": 0.4,
            "recovery_observation_seconds": 0.4,
            "suspicion_clear_seconds": 0.4,
            "medium_window_seconds": 0.3,
        })
        self.detector = FatigueDetector(self.config, clock=self.clock)
        self.detector.apply_calibration(personal_profile())
        self.feed()
        self.reach_critical()
        state, reason = self.feed_for(0.7, ear=0.32)
        self.assertEqual("RECUPERACION", state)
        self.assertIn("observacion posterior", self.detector.last_transition_reason)
        for _ in range(10):
            state, reason = self.feed(ear=0.32)
            if state == "SOSPECHA":
                break
        self.assertEqual("SOSPECHA", state)
        self.assertIn("descenso controlado", self.detector.last_transition_reason)
        for _ in range(10):
            state, reason = self.feed(ear=0.32)
            if state == "ALERTA":
                break
        self.assertEqual("ALERTA", state)
        self.assertIn("Estabilidad sostenida", self.detector.last_transition_reason)

    def test_13_relapse_during_recovery_returns_to_critical(self):
        self.config["fatigue"].update({
            "critical_minimum_hold_seconds": 0.3,
            "critical_open_recovery_seconds": 0.3,
            "recovery_relapse_closed_seconds": 0.7,
        })
        self.detector = FatigueDetector(self.config, clock=self.clock)
        self.detector.apply_calibration(personal_profile())
        self.feed()
        self.reach_critical()
        self.feed_for(0.6, ear=0.32)
        self.assertEqual("RECUPERACION", self.detector.state)
        state, reason = self.feed_for(0.9, ear=0.10)
        self.assertEqual("CRITICO", state)
        self.assertIn("recaida", self.detector.last_transition_reason.lower())

    def test_14_variable_fps_uses_elapsed_time_not_frame_count(self):
        state = "ALERTA"
        reason = ""
        for step, fps in ((0.30, 4.0), (0.05, 25.0), (0.40, 3.0),
                          (0.10, 15.0), (0.55, 2.0), (0.70, 20.0)):
            state, reason = self.feed(seconds=step, fps=fps, ear=0.10)
        state, reason = self.feed(seconds=0.1, fps=20.0, ear=0.10)
        self.assertEqual("CRITICO", state)
        self.assertIn("vigilancia", reason)

    def test_15_single_visible_eye_is_degraded_and_cannot_raise_critical(self):
        state, reason = self.feed_for(
            2.4, ear=0.10, left_reliable=True, right_reliable=False
        )
        self.assertEqual("VISION_DEGRADADA", self.last_metrics["vision_state"])
        self.assertEqual(1, self.last_metrics["valid_eye_count"])
        self.assertNotEqual("CRITICO", state)
        self.assertIn(state, ("SOSPECHA", "SOMNOLENCIA"))
        self.assertTrue(reason)

    def test_16_invalid_measurements_do_not_accumulate_perclos(self):
        self.feed(ear=0.10, left_reliable=False, right_reliable=False)
        before = self.last_metrics["perclos_valid_seconds"]
        self.feed_for(4.0, ear=0.10, left_reliable=False, right_reliable=False)
        self.assertAlmostEqual(before, self.last_metrics["perclos_valid_seconds"])
        self.assertEqual(0.0, self.last_metrics["perclos"])
        self.assertFalse(self.last_metrics["perclos_reliable"])

    def test_17_perclos_uses_valid_time_and_reports_coverage(self):
        self.config["fatigue"]["perclos_min_valid_seconds"] = 2.0
        self.feed_for(3.0, ear=0.10, left_reliable=False, right_reliable=False)
        self.feed_for(3.0, ear=0.32)
        self.assertGreater(self.last_metrics["perclos_total_seconds"], 5.0)
        self.assertLess(self.last_metrics["perclos_coverage"], 0.65)
        self.assertFalse(self.last_metrics["perclos_reliable"])

    def test_18_high_perclos_with_coverage_becomes_somnolence(self):
        self.config["fatigue"]["perclos_min_valid_seconds"] = 2.0
        for _ in range(12):
            self.feed_for(0.35, ear=0.10)
            self.feed_for(0.55, ear=0.32)
        self.assertTrue(self.last_metrics["perclos_reliable"])
        self.assertGreaterEqual(self.last_metrics["perclos"], 0.25)
        self.assertIn(self.detector.state, ("SOSPECHA", "SOMNOLENCIA"))
        self.assertTrue(self.detector.reason)

    def test_19_isolated_head_nod_is_weak(self):
        self.config["fatigue"]["head_nod_min_seconds"] = 0.35
        self.detector = FatigueDetector(self.config, clock=self.clock)
        self.detector.apply_calibration(personal_profile())
        self.feed()
        self.feed(seconds=0.2, pitch=0.0)
        self.feed(seconds=0.2, pitch=25.0)
        self.feed(seconds=0.2, pitch=25.0)
        state, reason = self.feed(seconds=0.2, pitch=0.0)
        self.assertEqual(1, self.last_metrics["recent_nods"])
        self.assertEqual("ALERTA", state)
        self.assertTrue(reason)

    def test_20_yawn_and_prolonged_blink_are_multimodal_suspicion(self):
        self.yawn()
        state, reason = self.blink(0.70)
        self.assertEqual("SOSPECHA", state)
        self.assertIn("multimodales", reason)

    def test_21_fast_alternation_does_not_oscillate_state(self):
        states = []
        for _ in range(12):
            states.append(self.feed(seconds=0.1, ear=0.22)[0])
            states.append(self.feed(seconds=0.1, ear=0.32)[0])
        self.assertEqual({"ALERTA"}, set(states))
        self.assertTrue(self.detector.reason)

    def test_22_duplicate_frame_sequence_does_not_duplicate_time(self):
        sample = driver_metrics(ear=0.10, sequence=10)
        self.detector.update(sample, analysis_fps=20.0)
        self.clock.advance(1.0)
        repeated = driver_metrics(ear=0.10, sequence=10)
        state, reason = self.detector.update(repeated, analysis_fps=20.0)
        self.assertEqual(0.0, repeated["current_closure_seconds"])
        self.assertNotEqual("CRITICO", state)
        self.assertTrue(reason)

    def test_23_paused_period_is_excluded_from_perclos_window(self):
        self.feed_for(2.0, ear=0.32)
        before = self.last_metrics["perclos_total_seconds"]
        for _ in range(10):
            self.feed(seconds=0.5, paused=True, ear=0.10)
        self.feed(ear=0.32)
        after = self.last_metrics["perclos_total_seconds"]
        self.assertLess(after - before, 1.0)
        self.assertTrue(self.detector.reason)

    def test_24_initializing_vision_requires_configured_valid_time(self):
        config = default_config()
        config["vision_reliability"]["initial_valid_seconds"] = 0.5
        detector = FatigueDetector(config, clock=self.clock)
        detector.apply_calibration(personal_profile())
        first = driver_metrics()
        state, reason = detector.update(first, analysis_fps=20.0)
        self.assertEqual("INICIALIZANDO", first["vision_state"])
        self.clock.advance(0.6)
        second = driver_metrics()
        state, reason = detector.update(second, analysis_fps=20.0)
        self.assertEqual("VISION_VALIDA", second["vision_state"])
        self.assertEqual("ALERTA", state)
        self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
