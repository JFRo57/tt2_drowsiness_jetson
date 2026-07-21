import time


class AlertController(object):
    LEVEL_1_FREQ = 2500
    LEVEL_2_FREQ = 3500
    LEVEL_3_FREQ = 4500

    def __init__(self, gpio_controller):
        self.gpio = gpio_controller
        self.state = "INICIALIZANDO"
        self.last_state = None
        self.start_time = time.monotonic()

    def update(self, fatigue_state, mode):
        now = time.monotonic()
        if fatigue_state != self.last_state:
            self.start_time = now
            self.last_state = fatigue_state
        t = now - self.start_time
        if mode == "EMERGENCY" or fatigue_state == "PARO_EMERGENCIA":
            self.gpio.set_buzzer_tone(None)
            self.gpio.set_led_state(False, False, True)
            return
        if mode == "MAINTENANCE" or fatigue_state == "MANTENIMIENTO":
            self.gpio.set_buzzer_tone(None)
            self.gpio.set_led_state(False, True, False)
            return
        led = self._led_pattern(fatigue_state, t)
        tone = self._buzzer_tone(fatigue_state, t)
        self.gpio.set_led_state(*led)
        self.gpio.set_buzzer_tone(tone)

    def _led_pattern(self, state, t):
        if state == "INICIALIZANDO":
            phase = int(t * 3) % 3
            return phase == 0, phase == 1, phase == 2
        if state in ("NORMAL", "PARPADEO"):
            return True, False, False
        if state == "POSIBLE_SOMNOLENCIA":
            return False, int(t * 1.2) % 2 == 0, False
        if state == "ROSTRO_NO_DETECTADO":
            return False, int(t * 3.0) % 3 != 1, False
        if state == "ALERTA":
            return False, False, int(t * 3.0) % 2 == 0
        if state == "ALERTA_CRITICA":
            return False, False, int(t * 7.0) % 2 == 0
        if state == "ERROR":
            phase = t % 1.2
            return False, False, phase < 0.15 or 0.3 < phase < 0.45
        return False, False, False

    def _buzzer_tone(self, state, t):
        if state == "POSIBLE_SOMNOLENCIA":
            return self._level_1_tone(t)
        if state == "ALERTA":
            return self._level_2_tone(t)
        if state == "ALERTA_CRITICA":
            return self._level_3_tone(t)
        return None

    def _level_1_tone(self, t):
        # Low urgency: sparse pulses, easy to notice but not aggressive.
        # 2500 Hz, 200 ms ON / 1000 ms OFF, 3 repetitions.
        if t >= 3.0:
            return None
        phase = t % 1.2
        return self.LEVEL_1_FREQ if phase < 0.200 else None

    def _level_2_tone(self, t):
        # Medium urgency: regular grouped pulses, repeated while alert persists.
        # 3500 Hz, 200 ms ON / 300 ms OFF, 3 repetitions, 900 ms pause.
        phase = t % 2.4
        if phase >= 1.5:
            return None
        beat = phase % 0.5
        return self.LEVEL_2_FREQ if beat < 0.200 else None

    def _level_3_tone(self, t):
        # High urgency: faster 5-pulse burst.
        # 4500 Hz, 150 ms ON / 150 ms OFF, 5 repetitions, 600 ms pause.
        cycle = t % 2.1
        if cycle >= 1.5:
            return None
        beat = cycle % 0.3
        return self.LEVEL_3_FREQ if beat < 0.150 else None
