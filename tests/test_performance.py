import unittest

import numpy as np

from src.application import default_config
from src.camera import CameraManager
from src.face_analyzer import FaceAnalyzer
from src.fatigue_detector import FatigueDetector
from src.gpio_controller import GPIOController
from src.presentation_ui import PresentationUI


class FakeGPIO(object):
    def __init__(self):
        self.outputs = []

    def output(self, pin, level):
        self.outputs.append((pin, level))


class LatestFrameTests(unittest.TestCase):
    def test_wait_returns_only_new_frame_without_copy(self):
        camera = CameraManager(default_config()["camera"])
        published = np.zeros((4, 6, 3), dtype=np.uint8)
        with camera.condition:
            camera.latest_frame = published
            camera.latest_time = 12.5
            camera.frame_sequence = 7

        frame, timestamp, sequence = camera.wait_for_frame(
            after_sequence=6,
            timeout=0.01,
            copy=False,
        )
        self.assertIs(published, frame)
        self.assertEqual(12.5, timestamp)
        self.assertEqual(7, sequence)

        frame, timestamp, sequence = camera.wait_for_frame(
            after_sequence=7,
            timeout=0.01,
            copy=False,
        )
        self.assertIsNone(frame)
        self.assertEqual(7, sequence)


class VisionSchedulingTests(unittest.TestCase):
    def test_no_face_detector_is_spaced(self):
        analyzer = object.__new__(FaceAnalyzer)
        analyzer.last_rect = None
        analyzer.frame_index = 1
        analyzer.no_face_detection_interval = 3
        analyzer.last_detection_source = "inicial"
        analyzer._detect_face = lambda gray: "detectado"
        gray = np.zeros((10, 10), dtype=np.uint8)

        self.assertEqual("detectado", analyzer._locate_face(gray))
        analyzer.frame_index = 2
        self.assertIsNone(analyzer._locate_face(gray))
        self.assertEqual("busqueda_espaciada", analyzer.last_detection_source)
        analyzer.frame_index = 3
        self.assertEqual("detectado", analyzer._locate_face(gray))

    def test_v2_defaults_preserve_hardware_and_enable_fast_path(self):
        config = default_config()
        self.assertEqual(33, config["buzzer"]["board_pin"])
        self.assertEqual("pwm_native", config["buzzer"]["type"])
        self.assertEqual([29, 31, 32], [
            config["leds"]["green_board_pin"],
            config["leds"]["yellow_board_pin"],
            config["leds"]["red_board_pin"],
        ])
        self.assertEqual("landmarks", config["dlib"]["tracking_mode"])
        self.assertEqual(0.5, config["dlib"]["detector_scale"])
        self.assertEqual(20.0, config["performance"]["target_analysis_fps"])


class PerclosTests(unittest.TestCase):
    def setUp(self):
        config = default_config()
        config["fatigue"]["perclos_window_seconds"] = 60.0
        self.detector = FatigueDetector(config)

    def test_incremental_perclos_matches_elapsed_time(self):
        self.detector._update_perclos(0.0, False)
        self.detector._update_perclos(2.0, True)
        self.detector._update_perclos(4.0, False)
        self.assertAlmostEqual(0.5, self.detector.current_perclos(4.0))

        self.detector._update_perclos(62.0, False)
        self.assertAlmostEqual(2.0 / 60.0, self.detector.current_perclos(62.0))
        self.detector._update_perclos(64.0, False)
        self.assertAlmostEqual(0.0, self.detector.current_perclos(64.0))

    def test_face_loss_does_not_count_unknown_interval(self):
        self.detector._update_perclos(0.0, True)
        self.detector._update_perclos(2.0, True)
        self.detector._pause_perclos(2.0)
        self.detector._update_perclos(20.0, False)

        self.assertAlmostEqual(1.0, self.detector.current_perclos(20.0))
        self.detector._update_perclos(80.0, False)
        self.assertAlmostEqual(0.0, self.detector.current_perclos(80.0))


class GPIOWriteCoalescingTests(unittest.TestCase):
    def test_identical_led_state_is_not_written_twice(self):
        config = default_config()
        config["gpio"]["enabled"] = True
        config["gpio"]["simulation_mode"] = False
        controller = GPIOController(config)
        controller.simulation_mode = False
        controller.GPIO = FakeGPIO()

        controller.set_led_state(True, False, False)
        controller.set_led_state(True, False, False)
        self.assertEqual(3, len(controller.GPIO.outputs))

        controller.set_led_state(False, True, False)
        self.assertEqual(6, len(controller.GPIO.outputs))


class InterfaceAllocationTests(unittest.TestCase):
    def test_information_panel_reuses_canvas(self):
        config = default_config()
        ui = PresentationUI(config)
        frame = np.zeros((100, 160, 3), dtype=np.uint8)

        class Detector(object):
            state = "NORMAL"
            reason = "prueba"
            ear_threshold = 0.22
            drowsy_ear_threshold = 0.26

        class Mode(object):
            mode = "AUTOMATIC"
            source = "simulado"

        class GPIO(object):
            leds = {"green": True, "yellow": False, "red": False}
            buzzer_on = False
            buzzer_frequency = 0
            error = None

        first = ui._add_panel(
            frame, {}, Detector(), Mode(), GPIO(), 30.0, 20.0, "GPIO simulado"
        )
        second = ui._add_panel(
            frame, {}, Detector(), Mode(), GPIO(), 30.0, 20.0, "GPIO simulado"
        )
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
