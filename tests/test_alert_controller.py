import unittest

from src.alert_controller import AlertController


class FakeClock(object):
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeGPIO(object):
    def __init__(self):
        self.leds = None
        self.tone = None

    def set_led_state(self, green, yellow, red):
        self.leds = (green, yellow, red)

    def set_buzzer_tone(self, tone):
        self.tone = tone


class AlertControllerTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.gpio = FakeGPIO()
        config = {
            "alerts": {
                "critical_frequency": 4400,
                "supervision_frequency": 2900,
                "critical_led_hz": 6.0,
                "supervision_led_hz": 2.0,
            },
        }
        self.alerts = AlertController(self.gpio, config, clock=self.clock)

    def test_supervision_failure_has_priority_over_fatigue(self):
        selected = self.alerts.update(
            "CRITICO", "AUTOMATIC", "CAMARA_OBSTRUIDA_O_FALLO"
        )

        self.assertEqual("SUPERVISION_FALLO", selected)
        self.assertEqual(2900, self.gpio.tone)

    def test_critical_replaces_lower_priority_pattern(self):
        self.alerts.update("SOSPECHA", "AUTOMATIC", "VISION_VALIDA")
        selected = self.alerts.update("CRITICO", "AUTOMATIC", "VISION_VALIDA")

        self.assertEqual("CRITICO", selected)
        self.assertEqual(4400, self.gpio.tone)
        self.assertEqual((False, False, True), self.gpio.leds)

    def test_recovery_has_no_buzzer(self):
        selected = self.alerts.update(
            "RECUPERACION", "AUTOMATIC", "VISION_VALIDA"
        )

        self.assertEqual("RECUPERACION", selected)
        self.assertIsNone(self.gpio.tone)


if __name__ == "__main__":
    unittest.main()
