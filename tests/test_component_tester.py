import unittest

from scripts.test_components import ComponentTester


class FakeClock(object):
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += float(seconds)


class FakeHardwareGPIO(object):
    def __init__(self):
        self.levels = {29: 0, 31: 0, 32: 0}

    def input(self, pin):
        return self.levels[pin]


class FakeController(object):
    def __init__(self):
        self.GPIO = FakeHardwareGPIO()
        self.led_cfg = {
            "active_high": True,
            "green_board_pin": 29,
            "yellow_board_pin": 31,
            "red_board_pin": 32,
        }
        self.buzzer_cfg = {"board_pin": 33}
        self.error = None
        self.led_history = []
        self.tone_history = []

    def set_led_state(self, green=False, yellow=False, red=False):
        state = (bool(green), bool(yellow), bool(red))
        self.led_history.append(state)
        self.GPIO.levels[29] = int(state[0])
        self.GPIO.levels[31] = int(state[1])
        self.GPIO.levels[32] = int(state[2])

    def set_buzzer_tone(self, frequency=None):
        self.tone_history.append(frequency)
        return True

    def all_outputs_off(self):
        self.set_buzzer_tone(None)
        self.set_led_state(False, False, False)

    @staticmethod
    def buzzer_backend():
        return "simulado para prueba"


class ComponentTesterTests(unittest.TestCase):
    def _tester(self, controller, clock):
        return ComponentTester(
            controller,
            led_seconds=0.1,
            tone_seconds=0.1,
            pattern_seconds=0.1,
            update_hz=10.0,
            pause_seconds=0.0,
            output=lambda message: None,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    def test_led_sequence_activates_each_output_individually(self):
        controller = FakeController()
        tester = self._tester(controller, FakeClock())

        tester.run_led_tests()

        self.assertIn((True, False, False), controller.led_history)
        self.assertIn((False, True, False), controller.led_history)
        self.assertIn((False, False, True), controller.led_history)
        self.assertEqual((False, False, False), controller.led_history[-1])

    def test_buzzer_sequence_uses_the_three_study_frequencies(self):
        controller = FakeController()
        tester = self._tester(controller, FakeClock())

        tester.run_buzzer_tests()

        tones = [value for value in controller.tone_history if value is not None]
        self.assertEqual([2500, 3500, 4500], tones)

    def test_pattern_sequence_reproduces_all_alert_levels(self):
        controller = FakeController()
        tester = self._tester(controller, FakeClock())

        tester.run_pattern_tests()

        self.assertIn((True, False, False), controller.led_history)
        self.assertIn((False, True, False), controller.led_history)
        self.assertIn((False, False, True), controller.led_history)
        self.assertIn(2500, controller.tone_history)
        self.assertIn(3500, controller.tone_history)
        self.assertIn(4500, controller.tone_history)


if __name__ == "__main__":
    unittest.main()
