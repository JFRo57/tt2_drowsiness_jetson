import os
import shutil
import tempfile
import unittest

from src.gpio_controller import GPIOController


def make_config(sysfs_root, buzzer_type="pwm_native"):
    return {
        "gpio": {"enabled": True, "simulation_mode": False},
        "buzzer": {
            "enabled": True,
            "type": buzzer_type,
            "board_pin": 33,
            "active_high": False,
            "muted": False,
            "idle_frequency": 3000,
            "pwm_duty_cycle": 50,
            "pwm_chip": 0,
            "pwm_channel": 2,
            "pwm_sysfs_root": sysfs_root,
            "min_frequency": 2000,
            "max_frequency": 5000,
        },
        "leds": {"enabled": False},
        "switch": {"enabled": False},
    }


class FakeGPIO(object):
    def __init__(self):
        self.outputs = []
        self.cleaned = False

    def output(self, pin, level):
        self.outputs.append((pin, level))

    def cleanup(self):
        self.cleaned = True


class GPIOControllerBuzzerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        chip = os.path.join(self.tempdir, "pwmchip0")
        pwm = os.path.join(chip, "pwm2")
        os.makedirs(pwm)
        self._write(os.path.join(chip, "export"), "")
        self._write(os.path.join(pwm, "period"), "333333")
        self._write(os.path.join(pwm, "duty_cycle"), "333333")
        self._write(os.path.join(pwm, "enable"), "1")
        self._write(os.path.join(pwm, "polarity"), "normal")

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    @staticmethod
    def _write(path, value):
        with open(path, "w") as stream:
            stream.write(value)

    @staticmethod
    def _read(path):
        with open(path, "r") as stream:
            return stream.read()

    def _path(self, name):
        return os.path.join(self.tempdir, "pwmchip0", "pwm2", name)

    def _controller(self, buzzer_type="pwm_native", sysfs_root=None):
        config = make_config(sysfs_root or self.tempdir, buzzer_type)
        controller = GPIOController(config)
        controller.simulation_mode = False
        controller.GPIO = FakeGPIO()
        return controller

    def test_native_pwm_generates_tone_at_fifty_percent(self):
        controller = self._controller()

        self.assertTrue(controller.set_buzzer_tone(2500))

        self.assertEqual("400000", self._read(self._path("period")))
        self.assertEqual("200000", self._read(self._path("duty_cycle")))
        self.assertEqual("1", self._read(self._path("enable")))

    def test_native_pwm_off_is_high_and_remains_enabled(self):
        controller = self._controller()
        controller.set_buzzer_tone(2500)

        self.assertTrue(controller.set_buzzer_tone(None))

        self.assertEqual("400000", self._read(self._path("duty_cycle")))
        self.assertEqual("1", self._read(self._path("enable")))

    def test_native_pwm_changes_frequency_without_disabling(self):
        controller = self._controller()
        controller.set_buzzer_tone(2500)

        self.assertTrue(controller.set_buzzer_tone(4000))

        self.assertEqual("250000", self._read(self._path("period")))
        self.assertEqual("125000", self._read(self._path("duty_cycle")))
        self.assertEqual("1", self._read(self._path("enable")))

    def test_native_pwm_reasserts_requested_tone_periodically(self):
        controller = self._controller()
        controller.set_buzzer_tone(2500)
        self._write(self._path("duty_cycle"), "400000")
        controller._last_native_buzzer_output_at -= (
            controller.output_refresh_seconds + 0.1
        )

        controller.set_buzzer_tone(2500)

        self.assertEqual("200000", self._read(self._path("duty_cycle")))

    def test_cleanup_leaves_low_trigger_module_high(self):
        controller = self._controller()
        controller.set_buzzer_tone(3500)

        controller.cleanup()

        self.assertEqual(self._read(self._path("period")), self._read(self._path("duty_cycle")))
        self.assertEqual("1", self._read(self._path("enable")))

    def test_native_pwm_failure_is_reported_and_latched(self):
        missing_root = os.path.join(self.tempdir, "missing")
        controller = self._controller(sysfs_root=missing_root)

        self.assertFalse(controller.set_buzzer_tone(3000))
        self.assertFalse(controller.set_buzzer_tone(3000))

        self.assertTrue(controller.buzzer_faulted)
        self.assertFalse(controller.buzzer_on)
        self.assertIn("PWM nativo", controller.error)

    def test_active_buzzer_uses_active_low_digital_output(self):
        controller = self._controller(buzzer_type="active")

        self.assertTrue(controller.set_buzzer_tone(3000))
        self.assertTrue(controller.set_buzzer_tone(None))

        self.assertEqual([(33, 0), (33, 1)], controller.GPIO.outputs)


if __name__ == "__main__":
    unittest.main()
