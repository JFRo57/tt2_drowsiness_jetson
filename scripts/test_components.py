#!/usr/bin/env python3
"""Prueba fisica independiente de LEDs y buzzer, sin camara ni detector."""

import argparse
import os
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import load_config
from src.alert_controller import AlertController
from src.gpio_controller import GPIOController


class ComponentTester(object):
    LED_CASES = (
        ("verde", (True, False, False), "green_board_pin"),
        ("amarillo", (False, True, False), "yellow_board_pin"),
        ("rojo", (False, False, True), "red_board_pin"),
    )
    TONE_CASES = (
        ("nivel 1", AlertController.LEVEL_1_FREQ),
        ("nivel 2", AlertController.LEVEL_2_FREQ),
        ("nivel 3", AlertController.LEVEL_3_FREQ),
    )
    PATTERN_CASES = (
        ("NORMAL", 2.0),
        ("POSIBLE_SOMNOLENCIA", 3.2),
        ("ALERTA", 4.8),
        ("ALERTA_CRITICA", 4.2),
    )

    def __init__(
        self,
        gpio,
        led_seconds=2.0,
        tone_seconds=1.5,
        pattern_seconds=None,
        update_hz=50.0,
        pause_seconds=0.35,
        output=print,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ):
        self.gpio = gpio
        self.led_seconds = max(0.0, float(led_seconds))
        self.tone_seconds = max(0.0, float(tone_seconds))
        self.pattern_seconds = (
            None if pattern_seconds is None else max(0.0, float(pattern_seconds))
        )
        self.update_period = 1.0 / max(10.0, float(update_hz))
        self.pause_seconds = max(0.0, float(pause_seconds))
        self.output = output
        self.sleep = sleep
        self.monotonic = monotonic

    def _pause(self):
        if self.pause_seconds:
            self.sleep(self.pause_seconds)

    def _read_led_levels(self):
        if self.gpio.GPIO is None or not hasattr(self.gpio.GPIO, "input"):
            return "lectura no disponible"
        values = []
        for label, _, pin_key in self.LED_CASES:
            pin = int(self.gpio.led_cfg[pin_key])
            try:
                level = int(self.gpio.GPIO.input(pin))
                values.append("%s=BOARD%d:%s" % (label, pin, level))
            except Exception as exc:
                values.append("%s=ERROR(%s)" % (label, exc))
        return ", ".join(values)

    def run_led_tests(self):
        self.output("\n=== PRUEBA INDIVIDUAL DE LEDS ===")
        active_level = 1 if self.gpio.led_cfg.get("active_high", True) else 0
        self.output(
            "Polaridad configurada: %s; nivel de encendido=%d"
            % (
                "activa-alta" if active_level else "activa-baja",
                active_level,
            )
        )
        for label, state, pin_key in self.LED_CASES:
            pin = int(self.gpio.led_cfg[pin_key])
            self.output(
                "LED %s ON: BOARD %d durante %.1f s"
                % (label, pin, self.led_seconds)
            )
            self.gpio.set_led_state(*state)
            self.output("Lectura GPIO: %s" % self._read_led_levels())
            self.sleep(self.led_seconds)
            self.gpio.set_led_state(False, False, False)
            self._pause()

    def run_buzzer_tests(self):
        self.output("\n=== PRUEBA INDIVIDUAL DEL BUZZER ===")
        self.output(
            "Backend: %s; BOARD %s"
            % (self.gpio.buzzer_backend(), self.gpio.buzzer_cfg["board_pin"])
        )
        for label, frequency in self.TONE_CASES:
            self.output(
                "Buzzer %s: %d Hz durante %.1f s"
                % (label, frequency, self.tone_seconds)
            )
            if not self.gpio.set_buzzer_tone(frequency):
                raise RuntimeError(
                    self.gpio.error or "No se pudo activar el buzzer"
                )
            self.sleep(self.tone_seconds)
            self.gpio.set_buzzer_tone(None)
            self._pause()

    def _drive_pattern(self, state, duration):
        alerts = AlertController(self.gpio)
        started = self.monotonic()
        while True:
            elapsed = self.monotonic() - started
            if elapsed >= duration - 1e-9:
                break
            led_state = alerts._led_pattern(state, elapsed)
            tone = alerts._buzzer_tone(state, elapsed)
            self.gpio.set_led_state(*led_state)
            if not self.gpio.set_buzzer_tone(tone):
                raise RuntimeError(
                    self.gpio.error or "No se pudo actualizar el buzzer"
                )
            remaining = duration - (self.monotonic() - started)
            if remaining > 0:
                self.sleep(min(self.update_period, remaining))
        self.gpio.all_outputs_off()

    def run_pattern_tests(self):
        self.output("\n=== PRUEBA DE PATRONES REALES ===")
        for state, default_duration in self.PATTERN_CASES:
            duration = (
                default_duration
                if self.pattern_seconds is None
                else self.pattern_seconds
            )
            self.output("Patron %-21s durante %.1f s" % (state, duration))
            self._drive_pattern(state, duration)
            self._pause()

    def run(self, section="all"):
        if section in ("all", "leds"):
            self.run_led_tests()
        if section in ("all", "buzzer"):
            self.run_buzzer_tests()
        if section in ("all", "patterns"):
            self.run_pattern_tests()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Prueba LEDs, tonos y patrones de alerta sin utilizar camara."
        )
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Ruta al archivo JSON de configuracion",
    )
    parser.add_argument(
        "--section",
        choices=("all", "leds", "buzzer", "patterns"),
        default="all",
        help="Seccion que se desea probar",
    )
    parser.add_argument(
        "--led-seconds",
        type=float,
        default=2.0,
        help="Duracion de encendido de cada LED",
    )
    parser.add_argument(
        "--tone-seconds",
        type=float,
        default=1.5,
        help="Duracion de cada tono continuo",
    )
    parser.add_argument(
        "--pattern-seconds",
        type=float,
        help="Duracion comun de cada patron; se omite para usar ciclos completos",
    )
    polarity = parser.add_mutually_exclusive_group()
    polarity.add_argument(
        "--led-active-high",
        action="store_true",
        help="Forzar LEDs activos con nivel HIGH",
    )
    polarity.add_argument(
        "--led-active-low",
        action="store_true",
        help="Forzar LEDs activos con nivel LOW",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    config = load_config(args.config)
    config["gpio"]["enabled"] = True
    config["gpio"]["simulation_mode"] = False
    config["switch"]["enabled"] = False
    config["buzzer"]["muted"] = False
    if args.led_active_high:
        config["leds"]["active_high"] = True
    elif args.led_active_low:
        config["leds"]["active_high"] = False

    gpio = GPIOController(config, force_simulation=False)
    if not gpio.setup():
        print("ERROR:", gpio.error or "No se pudo inicializar GPIO")
        return 1
    if gpio.simulation_mode or gpio.GPIO is None:
        print("ERROR: GPIO quedo en modo simulado; no se probaran componentes.")
        gpio.cleanup()
        return 1

    print("GPIO fisico inicializado")
    print(
        "LEDs: verde=BOARD%s, amarillo=BOARD%s, rojo=BOARD%s"
        % (
            gpio.led_cfg["green_board_pin"],
            gpio.led_cfg["yellow_board_pin"],
            gpio.led_cfg["red_board_pin"],
        )
    )
    print(
        "Buzzer: BOARD%s, %s"
        % (gpio.buzzer_cfg["board_pin"], gpio.buzzer_backend())
    )
    if gpio.error:
        print("ADVERTENCIA inicial:", gpio.error)

    tester = ComponentTester(
        gpio,
        led_seconds=args.led_seconds,
        tone_seconds=args.tone_seconds,
        pattern_seconds=args.pattern_seconds,
    )
    try:
        tester.run(args.section)
        if gpio.error:
            print("ERROR detectado:", gpio.error)
            return 1
        print("\nPrueba terminada. Todas las salidas quedan apagadas.")
        print(
            "El resultado electrico debe confirmarse visualmente o con multimetro."
        )
        return 0
    except KeyboardInterrupt:
        print("\nPrueba interrumpida por el usuario.")
        return 130
    except Exception as exc:
        print("\nERROR durante la prueba:", exc)
        return 1
    finally:
        gpio.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
