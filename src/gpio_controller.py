import os
import threading
import time

from .mode_controller import ModeController


class GPIOController(object):
    def __init__(self, config, force_simulation=False):
        self.config = config
        self.gpio_cfg = config["gpio"]
        self.buzzer_cfg = config["buzzer"]
        self.led_cfg = config["leds"]
        self.switch_cfg = config["switch"]
        self.enabled = bool(self.gpio_cfg.get("enabled", False)) and not force_simulation
        self.simulation_mode = bool(self.gpio_cfg.get("simulation_mode", True)) or force_simulation or not self.enabled
        self.GPIO = None
        self.lock = threading.Lock()
        self.buzzer_on = False
        self.buzzer_frequency = 0
        self.buzzer_pwm = None
        self.buzzer_pwm_started = False
        self.native_buzzer_frequency = 0
        self.native_pwm_ready = False
        self.native_buzzer_output_on = None
        self.buzzer_faulted = False
        self.buzzer_thread = None
        self.buzzer_stop = threading.Event()
        self.software_buzzer_frequency = 0
        self.leds = {"green": False, "yellow": False, "red": False}
        self._led_output_state = None
        self.simulated_mode = ModeController.AUTOMATIC
        self.mode_source = "simulado"
        self.error = None

    def setup(self):
        if self.simulation_mode or not self.enabled:
            return True
        try:
            import Jetson.GPIO as GPIO
            self.GPIO = GPIO
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            for pin in self._output_pins():
                GPIO.setup(pin, GPIO.OUT, initial=self._inactive_level(self.led_cfg.get("active_high", True)))
            if self.buzzer_cfg.get("enabled", True) and not self._uses_native_buzzer():
                GPIO.setup(int(self.buzzer_cfg["board_pin"]), GPIO.OUT, initial=self._inactive_level(self.buzzer_cfg.get("active_high", True)))
                # PWM nativo se controla mediante sysfs. No reclamar BOARD 33
                # como GPIO: en Tegra210 debe permanecer en funcion SFIO/PWM2.
            if self.switch_cfg.get("enabled", False):
                pud = GPIO.PUD_UP if self.switch_cfg.get("active_low", True) else GPIO.PUD_DOWN
                GPIO.setup(int(self.switch_cfg["auto_board_pin"]), GPIO.IN, pull_up_down=pud)
                GPIO.setup(int(self.switch_cfg["emergency_board_pin"]), GPIO.IN, pull_up_down=pud)
            self.all_outputs_off()
            self.mode_source = "fisico"
            return True
        except Exception as exc:
            self.error = "Error GPIO: %s" % exc
            self.enabled = False
            self.simulation_mode = True
            return False

    def _output_pins(self):
        if not self.led_cfg.get("enabled", True):
            return []
        return [int(self.led_cfg["green_board_pin"]), int(self.led_cfg["yellow_board_pin"]), int(self.led_cfg["red_board_pin"])]

    def _uses_software_buzzer(self):
        buzzer_type = str(self.buzzer_cfg.get("type", "passive_3pin")).lower()
        return buzzer_type in ("software_pwm",)

    def _uses_native_buzzer(self):
        buzzer_type = str(self.buzzer_cfg.get("type", "passive_3pin")).lower()
        return buzzer_type in ("passive", "passive_3pin", "pwm", "pwm_native")

    def _uses_active_buzzer(self):
        buzzer_type = str(self.buzzer_cfg.get("type", "passive_3pin")).lower()
        return buzzer_type in ("active", "active_3pin", "digital", "on_off")

    def buzzer_backend(self):
        if self._uses_software_buzzer():
            return "PWM por software (no recomendado con vision activa)"
        if self._uses_native_buzzer():
            return "PWM nativo"
        if self._uses_active_buzzer():
            return "digital ON/OFF"
        return "desconocido"

    @staticmethod
    def _active_level(active_high):
        return 1 if active_high else 0

    @staticmethod
    def _inactive_level(active_high):
        return 0 if active_high else 1

    def all_outputs_off(self):
        self.set_buzzer_state(False)
        self.set_led_state(False, False, False)

    def set_buzzer_state(self, on):
        default_frequency = int(self.buzzer_cfg.get("idle_frequency", 1000))
        self.set_buzzer_tone(default_frequency if on else None)

    def set_buzzer_tone(self, frequency=None):
        with self.lock:
            muted = bool(self.buzzer_cfg.get("muted", False))
            enabled = bool(self.buzzer_cfg.get("enabled", True))
            on = frequency is not None and float(frequency) > 0 and not muted and enabled
            self.buzzer_on = bool(on)
            self.buzzer_frequency = int(frequency) if on else 0
            if self.simulation_mode or self.GPIO is None or not enabled:
                return True
            if on and self.buzzer_faulted:
                self.buzzer_on = False
                self.buzzer_frequency = 0
                return False
            try:
                if self._uses_software_buzzer():
                    self._set_software_buzzer(on, self.buzzer_frequency)
                elif self._uses_native_buzzer():
                    self._set_passive_buzzer(on, self.buzzer_frequency)
                elif self._uses_active_buzzer():
                    level = self._active_level(self.buzzer_cfg.get("active_high", True)) if on else self._inactive_level(self.buzzer_cfg.get("active_high", True))
                    self.GPIO.output(int(self.buzzer_cfg["board_pin"]), level)
                else:
                    raise ValueError("tipo de buzzer no soportado: %s" % self.buzzer_cfg.get("type"))
                return True
            except Exception as exc:
                if self._uses_native_buzzer():
                    self.error = (
                        "Error buzzer PWM nativo en BOARD %s: %s. "
                        "Habilite PWM2 para el pin 33 con Jetson-IO y reinicie; "
                        "no se usara software_pwm automaticamente."
                    ) % (self.buzzer_cfg.get("board_pin"), exc)
                else:
                    self.error = "Error buzzer GPIO: %s" % exc
                self.buzzer_faulted = True
                self.buzzer_on = False
                self.buzzer_frequency = 0
                self._discard_native_pwm()
                return False

    def _set_software_buzzer(self, on, frequency):
        if not on:
            self._stop_software_buzzer()
            self.GPIO.output(int(self.buzzer_cfg["board_pin"]), self._inactive_level(self.buzzer_cfg.get("active_high", True)))
            return
        frequency = int(max(40, min(5000, int(frequency))))
        if self.buzzer_thread is not None and self.buzzer_thread.is_alive() and self.software_buzzer_frequency == frequency:
            return
        self._stop_software_buzzer()
        self.software_buzzer_frequency = frequency
        self.buzzer_stop.clear()
        self.buzzer_thread = threading.Thread(target=self._software_buzzer_loop, args=(frequency,))
        self.buzzer_thread.daemon = True
        self.buzzer_thread.start()

    def _stop_software_buzzer(self):
        if self.buzzer_thread is not None:
            self.buzzer_stop.set()
            self.buzzer_thread.join(0.2)
            self.buzzer_thread = None
        self.software_buzzer_frequency = 0

    def _software_buzzer_loop(self, frequency):
        pin = int(self.buzzer_cfg["board_pin"])
        active = self._active_level(self.buzzer_cfg.get("active_high", True))
        inactive = self._inactive_level(self.buzzer_cfg.get("active_high", True))
        half_period = 0.5 / float(frequency)
        next_toggle = time.perf_counter()
        level = active
        while not self.buzzer_stop.is_set():
            self.GPIO.output(pin, level)
            level = inactive if level == active else active
            next_toggle += half_period
            delay = next_toggle - time.perf_counter()
            if delay > 0.001:
                time.sleep(delay - 0.0005)
            while time.perf_counter() < next_toggle and not self.buzzer_stop.is_set():
                pass
        self.GPIO.output(pin, inactive)

    def _native_pwm_paths(self):
        root = str(self.buzzer_cfg.get("pwm_sysfs_root", "/sys/class/pwm"))
        chip = int(self.buzzer_cfg.get("pwm_chip", 0))
        channel = int(self.buzzer_cfg.get("pwm_channel", 2))
        chip_path = os.path.join(root, "pwmchip%d" % chip)
        pwm_path = os.path.join(chip_path, "pwm%d" % channel)
        return chip_path, pwm_path, channel

    @staticmethod
    def _write_pwm_value(path, value):
        with open(path, "w") as stream:
            stream.write(str(value))

    @staticmethod
    def _read_pwm_value(path):
        with open(path, "r") as stream:
            return int(stream.read().strip())

    def _ensure_native_pwm(self, frequency):
        chip_path, pwm_path, channel = self._native_pwm_paths()
        if not os.path.isdir(chip_path):
            raise RuntimeError("no existe %s" % chip_path)
        if not os.path.isdir(pwm_path):
            self._write_pwm_value(os.path.join(chip_path, "export"), channel)
            deadline = time.monotonic() + 1.0
            while not os.path.isdir(pwm_path) and time.monotonic() < deadline:
                time.sleep(0.01)
        if not os.path.isdir(pwm_path):
            raise RuntimeError("no se pudo exportar %s" % pwm_path)

        minimum = int(self.buzzer_cfg.get("min_frequency", 2000))
        maximum = int(self.buzzer_cfg.get("max_frequency", 5000))
        frequency = max(minimum, min(maximum, int(frequency)))
        period = int(1000000000.0 / float(frequency))
        period_path = os.path.join(pwm_path, "period")
        duty_path = os.path.join(pwm_path, "duty_cycle")
        enable_path = os.path.join(pwm_path, "enable")
        current_period = self._read_pwm_value(period_path)
        enabled = self._read_pwm_value(enable_path)

        if current_period != period:
            if current_period > 0:
                self._write_pwm_value(duty_path, 0)
            self._write_pwm_value(period_path, period)

        self.native_pwm_ready = True
        self.native_buzzer_frequency = frequency
        self.buzzer_pwm_started = bool(enabled)
        return pwm_path, period, enabled

    def _set_passive_buzzer(self, on, frequency):
        self._stop_software_buzzer()
        if not on:
            frequency = self.native_buzzer_frequency or int(self.buzzer_cfg.get("idle_frequency", 3000))
        minimum = int(self.buzzer_cfg.get("min_frequency", 2000))
        maximum = int(self.buzzer_cfg.get("max_frequency", 5000))
        frequency = max(minimum, min(maximum, int(frequency)))

        if (self.native_pwm_ready and self.native_buzzer_output_on == bool(on)
                and (not on or self.native_buzzer_frequency == frequency)):
            return

        pwm_path, period, enabled = self._ensure_native_pwm(frequency)
        if on:
            duty_percent = float(self.buzzer_cfg.get("pwm_duty_cycle", 50.0))
            duty_percent = max(1.0, min(99.0, duty_percent))
        else:
            duty_percent = 0.0 if self.buzzer_cfg.get("active_high", True) else 100.0
        duty = int(period * (duty_percent / 100.0))
        self._write_pwm_value(os.path.join(pwm_path, "duty_cycle"), duty)
        if not enabled:
            self._write_pwm_value(os.path.join(pwm_path, "enable"), 1)
        self.buzzer_pwm_started = True
        self.native_buzzer_output_on = bool(on)

    def _discard_native_pwm(self):
        if not self._uses_native_buzzer() or not self.native_pwm_ready:
            return
        try:
            pwm_path, period, enabled = self._ensure_native_pwm(
                self.native_buzzer_frequency or int(self.buzzer_cfg.get("idle_frequency", 3000))
            )
            inactive_percent = 0.0 if self.buzzer_cfg.get("active_high", True) else 100.0
            self._write_pwm_value(
                os.path.join(pwm_path, "duty_cycle"),
                int(period * (inactive_percent / 100.0)),
            )
            if not enabled:
                self._write_pwm_value(os.path.join(pwm_path, "enable"), 1)
            self.native_buzzer_output_on = False
            self.buzzer_pwm_started = True
        except Exception:
            pass

    def set_led_state(self, green=False, yellow=False, red=False):
        with self.lock:
            desired = (bool(green), bool(yellow), bool(red))
            self.leds = {
                "green": desired[0],
                "yellow": desired[1],
                "red": desired[2],
            }
            if self.simulation_mode or self.GPIO is None or not self.led_cfg.get("enabled", True):
                return
            if desired == self._led_output_state:
                return
            active_high = self.led_cfg.get("active_high", True)
            pins = {"green": int(self.led_cfg["green_board_pin"]), "yellow": int(self.led_cfg["yellow_board_pin"]), "red": int(self.led_cfg["red_board_pin"])}
            for name, pin in pins.items():
                level = self._active_level(active_high) if self.leds[name] else self._inactive_level(active_high)
                self.GPIO.output(pin, level)
            self._led_output_state = desired

    def read_switch_mode(self):
        if self.simulation_mode or self.GPIO is None or not self.switch_cfg.get("enabled", False):
            self.mode_source = "simulado"
            return self.simulated_mode
        active_low = self.switch_cfg.get("active_low", True)
        auto_raw = self.GPIO.input(int(self.switch_cfg["auto_board_pin"]))
        emerg_raw = self.GPIO.input(int(self.switch_cfg["emergency_board_pin"]))
        auto_active = (auto_raw == 0) if active_low else (auto_raw == 1)
        emerg_active = (emerg_raw == 0) if active_low else (emerg_raw == 1)
        self.mode_source = "fisico"
        if emerg_active:
            return ModeController.EMERGENCY
        if auto_active:
            return ModeController.AUTOMATIC
        return ModeController.MAINTENANCE

    def set_simulated_mode(self, mode):
        self.simulated_mode = mode
        self.mode_source = "simulado"

    def self_test(self):
        if not self.setup():
            print(self.error or "No se pudo inicializar GPIO")
            return 1
        if self.simulation_mode or self.GPIO is None:
            print(self.error or "GPIO esta en modo simulado; no se escribira a pines fisicos")
            return 1
        try:
            print("GPIO fisico inicializado")
            print("Backend buzzer: %s" % self.buzzer_backend())
            print("LED verde: BOARD %s" % self.led_cfg["green_board_pin"])
            self.set_led_state(True, False, False)
            time.sleep(2.0)
            print("LED amarillo: BOARD %s" % self.led_cfg["yellow_board_pin"])
            self.set_led_state(False, True, False)
            time.sleep(2.0)
            print("LED rojo: BOARD %s" % self.led_cfg["red_board_pin"])
            self.set_led_state(False, False, True)
            time.sleep(2.0)
            print("Buzzer: BOARD %s a 2500 Hz" % self.buzzer_cfg["board_pin"])
            self.set_led_state(False, False, False)
            if not self.set_buzzer_tone(2500):
                print(self.error or "No se pudo iniciar el buzzer")
                return 1
            time.sleep(2.0)
            self.set_buzzer_tone(None)
            time.sleep(0.2)
            print("Buzzer: BOARD %s a 4000 Hz" % self.buzzer_cfg["board_pin"])
            if not self.set_buzzer_tone(4000):
                print(self.error or "No se pudo cambiar el tono del buzzer")
                return 1
            time.sleep(2.0)
            self.set_buzzer_tone(None)
            print("Prueba GPIO terminada")
            return 0
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            self.all_outputs_off()
            self._stop_software_buzzer()
            self._discard_native_pwm()
            if self.GPIO is not None:
                self.GPIO.cleanup()
        finally:
            self.GPIO = None
            self._led_output_state = None
