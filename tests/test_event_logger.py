import csv
import os
import tempfile
import unittest

from src.event_logger import EventLogger


class FakeClock(object):
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class FakeGPIO(object):
    leds = {"green": False, "yellow": False, "red": True}
    buzzer_on = True
    buzzer_frequency = 4500


class EventLoggerTests(unittest.TestCase):
    def test_elementary_and_critical_events_are_logged_and_flushed(self):
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "events.csv")
            logger = EventLogger({"logging": {
                "enabled": True,
                "events_only": True,
                "path": path,
                "flush_interval_seconds": 100.0,
            }}, clock=clock)
            logger.open()
            try:
                logger.log_transition(
                    "AUTOMATIC", "ALERTA", "ALERTA", "vision inicial",
                    {"vision_state": "VISION_VALIDA"}, 20.0, FakeGPIO(),
                )
                clock.now += 0.1
                logger.log_transition(
                    "AUTOMATIC", "ALERTA", "ALERTA", "evento critico",
                    {
                        "vision_state": "VISION_VALIDA",
                        "new_events": ["CIERRE_GRAVE"],
                        "critical_evidence": ["cierre_ocular_critico"],
                    },
                    19.0,
                    FakeGPIO(),
                )
                with open(path, "r") as handle:
                    rows = list(csv.DictReader(handle))
            finally:
                logger.close()

        self.assertEqual(2, len(rows))
        self.assertEqual("CIERRE_GRAVE", rows[-1]["new_events"])
        self.assertEqual("4500", rows[-1]["buzzer_frequency"])

    def test_old_csv_header_is_not_mixed_with_new_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "events.csv")
            with open(path, "w") as handle:
                handle.write("old,header\n")
            logger = EventLogger({"logging": {
                "enabled": True, "events_only": True, "path": path,
            }})
            logger.open()
            try:
                self.assertTrue(logger.path.endswith("events_v2.csv"))
            finally:
                logger.close()


if __name__ == "__main__":
    unittest.main()
