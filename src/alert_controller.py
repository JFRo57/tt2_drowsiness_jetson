import time


class AlertController(object):
    """Unico arbitro de LEDs y buzzer, con prioridad de supervision."""

    LEGACY_STATES = {
        "NORMAL": "ALERTA",
        "PARPADEO": "ALERTA",
        "POSIBLE_SOMNOLENCIA": "SOSPECHA",
        "ALERTA_CRITICA": "CRITICO",
        "ROSTRO_NO_DETECTADO": "SUPERVISION_ROSTRO",
    }

    def __init__(self, gpio_controller, config=None, clock=None):
        self.gpio = gpio_controller
        self.config = (config or {}).get("alerts", {})
        self.clock = clock or time.monotonic
        self.last_alert = None
        self.active_alert = "INICIALIZANDO"
        self.start_time = self.clock()

    def update(self, fatigue_state, mode, vision_state=None):
        now = self.clock()
        alert = self._select_alert(fatigue_state, mode, vision_state)
        if alert != self.last_alert:
            self.start_time = now
            self.last_alert = alert
        self.active_alert = alert
        elapsed = now - self.start_time
        leds = self._led_pattern(alert, elapsed)
        tone = self._buzzer_tone(alert, elapsed)
        self.gpio.set_led_state(*leds)
        self.gpio.set_buzzer_tone(tone)
        return alert

    def _select_alert(self, fatigue_state, mode, vision_state):
        state = self.LEGACY_STATES.get(fatigue_state, fatigue_state)
        if mode == "EMERGENCY" or state == "PARO_EMERGENCIA":
            return "PARO_EMERGENCIA"
        if mode == "MAINTENANCE" or state == "MANTENIMIENTO":
            return "MANTENIMIENTO"
        if vision_state == "CAMARA_OBSTRUIDA_O_FALLO":
            return "SUPERVISION_FALLO"
        if vision_state == "ROSTRO_NO_VISIBLE":
            return "SUPERVISION_ROSTRO"
        if state == "CRITICO":
            return "CRITICO"
        if state == "SOMNOLENCIA":
            return "SOMNOLENCIA"
        if state == "SOSPECHA":
            return "SOSPECHA"
        if state == "RECUPERACION":
            return "RECUPERACION"
        if state == "CALIBRACION":
            return "CALIBRACION"
        if state == "ERROR":
            return "ERROR"
        if vision_state == "VISION_DEGRADADA":
            return "VISION_DEGRADADA"
        if state == "ALERTA":
            return "ALERTA"
        return "INICIALIZANDO"

    def _led_pattern(self, alert, elapsed):
        initial_hz = float(self.config.get("initial_led_hz", 3.0))
        if alert == "PARO_EMERGENCIA":
            return False, False, True
        if alert == "MANTENIMIENTO":
            return False, True, False
        if alert == "ALERTA":
            return True, False, False
        if alert in ("CALIBRACION", "VISION_DEGRADADA"):
            key = ("calibration_led_hz" if alert == "CALIBRACION"
                   else "degraded_vision_led_hz")
            return False, int(elapsed * float(self.config.get(key, 1.2))) % 2 == 0, False
        if alert == "SOSPECHA":
            hz = float(self.config.get("suspicion_led_hz", 1.8))
            return False, int(elapsed * hz) % 2 == 0, False
        if alert == "SOMNOLENCIA":
            hz = float(self.config.get("somnolence_led_hz", 3.0))
            return False, False, int(elapsed * hz) % 2 == 0
        if alert == "CRITICO":
            hz = float(self.config.get("critical_led_hz", 7.0))
            return False, False, int(elapsed * hz) % 2 == 0
        if alert == "RECUPERACION":
            hz = float(self.config.get("recovery_led_hz", 1.0))
            return False, int(elapsed * hz) % 2 == 0, int(elapsed * hz) % 2 == 1
        if alert == "SUPERVISION_ROSTRO":
            hz = float(self.config.get("supervision_led_hz", 3.0))
            return False, int(elapsed * hz) % 3 != 1, False
        if alert == "SUPERVISION_FALLO":
            hz = float(self.config.get("failure_led_hz", 3.0))
            phase = int(elapsed * hz) % 2
            return False, phase == 0, phase == 1
        if alert == "ERROR":
            cycle = float(self.config.get("error_led_cycle_seconds", 1.2))
            pulse = float(self.config.get("error_led_pulse_seconds", 0.15))
            gap = float(self.config.get("error_led_gap_seconds", 0.15))
            phase = elapsed % cycle
            second_start = pulse + gap
            return (False, False,
                    phase < pulse or second_start <= phase < second_start + pulse)
        phase = int(elapsed * initial_hz) % 3
        return phase == 0, phase == 1, phase == 2

    def _buzzer_tone(self, alert, elapsed):
        if alert in ("PARO_EMERGENCIA", "MANTENIMIENTO", "ALERTA",
                     "CALIBRACION", "VISION_DEGRADADA", "RECUPERACION"):
            return None
        if alert == "SOSPECHA":
            notice = float(self.config.get("suspicion_notice_seconds", 3.0))
            if elapsed >= notice:
                return None
            cycle = float(self.config.get("suspicion_cycle_seconds", 1.2))
            on_time = float(self.config.get("suspicion_on_seconds", 0.2))
            return (int(self.config.get("suspicion_frequency", 2500))
                    if elapsed % cycle < on_time else None)
        if alert == "SOMNOLENCIA":
            cycle = elapsed % float(self.config.get("somnolence_cycle_seconds", 2.4))
            if cycle >= float(self.config.get("somnolence_group_seconds", 1.5)):
                return None
            beat = cycle % float(self.config.get("somnolence_beat_seconds", 0.5))
            return (int(self.config.get("somnolence_frequency", 3500))
                    if beat < float(self.config.get("somnolence_on_seconds", 0.2))
                    else None)
        if alert == "CRITICO":
            cycle = elapsed % float(self.config.get("critical_cycle_seconds", 2.1))
            if cycle >= float(self.config.get("critical_group_seconds", 1.5)):
                return None
            beat = cycle % float(self.config.get("critical_beat_seconds", 0.3))
            return (int(self.config.get("critical_frequency", 4500))
                    if beat < float(self.config.get("critical_on_seconds", 0.15))
                    else None)
        if alert in ("SUPERVISION_ROSTRO", "SUPERVISION_FALLO", "ERROR"):
            cycle = float(self.config.get("supervision_cycle_seconds", 2.0))
            on_time = float(self.config.get("supervision_on_seconds", 0.25))
            return (int(self.config.get("supervision_frequency", 3000))
                    if elapsed % cycle < on_time else None)
        return None
