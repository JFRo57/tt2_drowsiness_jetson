import unittest

from src.application import DrowsinessApplication


class FakeGPIOController(object):
    def __init__(self, setup_result, simulation_mode, gpio_object, error=None):
        self.setup_result = setup_result
        self.simulation_mode = simulation_mode
        self.GPIO = gpio_object
        self.error = error

    def setup(self):
        return self.setup_result


class FakeShutdown(object):
    def __init__(self):
        self.requested = False


class FakeMode(object):
    mode = "AUTOMATIC"

    def update_from_switch(self, gpio):
        return self.mode


class FakeAlerts(object):
    def __init__(self):
        self.calls = []

    def update(self, state, mode, vision_state=None):
        self.calls.append((state, mode))


class FakeCamera(object):
    error = None

    def __init__(self, shutdown):
        self.shutdown = shutdown

    def wait_for_frame(self, last_sequence, timeout, copy):
        self.shutdown.requested = True
        return None, 0.0, last_sequence


class ApplicationOutputTests(unittest.TestCase):
    def test_physical_gpio_setup_failure_stops_startup(self):
        app = object.__new__(DrowsinessApplication)
        app.config = {
            "gpio": {"enabled": True, "simulation_mode": False},
        }
        app.simulation = False
        app.gpio = FakeGPIOController(False, True, None, "fallo GPIO")

        with self.assertRaisesRegex(RuntimeError, "fallo GPIO"):
            app._setup_outputs()

    def test_explicit_simulation_accepts_missing_gpio(self):
        app = object.__new__(DrowsinessApplication)
        app.config = {
            "gpio": {"enabled": False, "simulation_mode": True},
        }
        app.simulation = True
        app.gpio = FakeGPIOController(True, True, None)

        app._setup_outputs()

    def test_alert_outputs_refresh_while_camera_has_no_frame(self):
        app = object.__new__(DrowsinessApplication)
        app.shutdown = FakeShutdown()
        app.mode = FakeMode()
        app.gpio = object()
        app.detector = type("Detector", (), {"state": "ALERTA_CRITICA"})()
        app.alerts = FakeAlerts()
        app.camera = FakeCamera(app.shutdown)
        app.frame_wait_timeout = 0.1
        app.ui = None

        app._loop()

        self.assertEqual(
            [("ALERTA_CRITICA", "AUTOMATIC")],
            app.alerts.calls,
        )


if __name__ == "__main__":
    unittest.main()
