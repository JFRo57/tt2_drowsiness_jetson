import time
from collections import deque


class FatigueDetector(object):
    STATES = (
        "INICIALIZANDO", "CALIBRANDO", "NORMAL", "PARPADEO", "POSIBLE_SOMNOLENCIA",
        "ALERTA", "ALERTA_CRITICA", "ROSTRO_NO_DETECTADO", "MANTENIMIENTO",
        "PARO_EMERGENCIA", "ERROR"
    )

    def __init__(self, config):
        self.config = config["fatigue"]
        self.calibration_config = config.get("calibration", {})
        self.state = "INICIALIZANDO"
        self.previous_state = None
        self.reason = "Inicializando sistema"
        self.ear_threshold = float(self.config["ear_threshold"])
        self.drowsy_ear_threshold = float(self.config.get("drowsy_ear_threshold", self.ear_threshold + 0.04))
        self.profile_min_confidence = float(self.calibration_config.get("profile_min_confidence", 0.15))
        self.profile_warning_seconds = float(self.calibration_config.get("profile_warning_seconds", 0.8))
        self.calibrated_profile = "NO_CALIBRADO"
        self.calibrated_profile_confidence = 0.0
        self.calibrated_profile_since = None
        self.calibrated_profile_seconds = 0.0
        self.closed_since = None
        self.yawn_since = None
        self.nod_since = None
        self.gaze_away_since = None
        self.no_face_since = None
        self.last_open_time = time.monotonic()
        self.last_transition = time.monotonic()
        self.perclos = deque(maxlen=3000)
        self.blinks = deque(maxlen=120)
        self.yawns = deque(maxlen=20)
        self.closed_duration = 0.0
        self.last_blink_duration = 0.0
        self.possible_yawn = False
        self.possible_nod = False
        self.gaze_away_seconds = 0.0

    def reset(self):
        threshold = self.ear_threshold
        drowsy_threshold = self.drowsy_ear_threshold
        self.state = "NORMAL"
        self.previous_state = None
        self.reason = "Metricas temporales reiniciadas"
        self.closed_since = None
        self.yawn_since = None
        self.nod_since = None
        self.gaze_away_since = None
        self.no_face_since = None
        self.last_open_time = time.monotonic()
        self.last_transition = time.monotonic()
        self.perclos.clear()
        self.blinks.clear()
        self.yawns.clear()
        self.closed_duration = 0.0
        self.last_blink_duration = 0.0
        self.possible_yawn = False
        self.possible_nod = False
        self.gaze_away_seconds = 0.0
        self.ear_threshold = threshold
        self.drowsy_ear_threshold = drowsy_threshold
        self.calibrated_profile = "NO_CALIBRADO"
        self.calibrated_profile_confidence = 0.0
        self.calibrated_profile_since = None
        self.calibrated_profile_seconds = 0.0

    def apply_calibrated_threshold(self, value):
        self.ear_threshold = float(value)

    def apply_calibration(self, result):
        thresholds = result.get("thresholds", {}) if result else {}
        if thresholds.get("ear_threshold") is not None:
            self.ear_threshold = float(thresholds["ear_threshold"])
        elif result and result.get("ear_threshold") is not None:
            self.ear_threshold = float(result["ear_threshold"])
        if thresholds.get("drowsy_ear_threshold") is not None:
            self.drowsy_ear_threshold = float(thresholds["drowsy_ear_threshold"])

    def update(self, metrics, mode="AUTOMATIC"):
        now = time.monotonic()
        self.previous_state = self.state
        if mode == "EMERGENCY":
            return self._set("PARO_EMERGENCIA", "Paro de emergencia activo", now)
        if mode == "MAINTENANCE":
            return self._set("MANTENIMIENTO", "Modo mantenimiento: alerta auditiva suspendida", now)
        if not metrics.get("face_detected", False):
            if self.no_face_since is None:
                self.no_face_since = now
            no_face_time = now - self.no_face_since
            metrics["no_face_seconds"] = no_face_time
            self.closed_since = None
            self.closed_duration = 0.0
            self.calibrated_profile = "DESCONOCIDO"
            self.calibrated_profile_confidence = 0.0
            self.calibrated_profile_since = None
            self.calibrated_profile_seconds = 0.0
            if no_face_time >= float(self.config["no_face_warning_seconds"]):
                return self._set("ROSTRO_NO_DETECTADO", "Rostro no detectado por %.1f s" % no_face_time, now)
            return self._set("NORMAL", "Perdida momentanea de rostro", now)
        self.no_face_since = None
        metrics["no_face_seconds"] = 0.0
        ear = metrics.get("ear")
        quality = float(metrics.get("quality", 0.0))
        reliable = ear is not None and quality >= 0.25
        closed = reliable and ear < self.ear_threshold
        self._update_perclos(now, bool(closed and reliable))
        self._update_eye_timing(now, closed, reliable)
        self._update_calibrated_profile(now, metrics)
        self._update_yawn(now, metrics)
        self._update_head(now, metrics)
        self._update_gaze(now, metrics)
        metrics.update({
            "ear_threshold": self.ear_threshold,
            "closed_seconds": self.closed_duration,
            "perclos": self.current_perclos(now),
            "blink_count_recent": len(self.blinks),
            "last_blink_seconds": self.last_blink_duration,
            "possible_yawn": self.possible_yawn,
            "recent_yawns": len(self.yawns),
            "possible_nod": self.possible_nod,
            "gaze_away_seconds": self.gaze_away_seconds,
            "calibrated_profile": self.calibrated_profile,
            "calibrated_profile_confidence": self.calibrated_profile_confidence,
            "calibrated_profile_seconds": self.calibrated_profile_seconds,
            "drowsy_ear_threshold": self.drowsy_ear_threshold,
        })
        return self._decide(now, metrics, closed, reliable)

    def _update_perclos(self, now, closed):
        self.perclos.append((now, closed))
        win = float(self.config["perclos_window_seconds"])
        while self.perclos and now - self.perclos[0][0] > win:
            self.perclos.popleft()

    def current_perclos(self, now):
        if len(self.perclos) < 2:
            return 0.0
        total = 0.0
        closed_total = 0.0
        prev_t, prev_closed = self.perclos[0]
        for t, closed in list(self.perclos)[1:]:
            dt = max(0.0, t - prev_t)
            total += dt
            if prev_closed:
                closed_total += dt
            prev_t, prev_closed = t, closed
        return closed_total / total if total > 0 else 0.0

    def _update_eye_timing(self, now, closed, reliable):
        if not reliable:
            return
        if closed:
            if self.closed_since is None:
                self.closed_since = now
            self.closed_duration = now - self.closed_since
        else:
            if self.closed_since is not None:
                dur = now - self.closed_since
                if float(self.config["blink_min_seconds"]) <= dur <= float(self.config["blink_max_seconds"]):
                    self.blinks.append(now)
                    self.last_blink_duration = dur
            self.closed_since = None
            self.closed_duration = 0.0
            self.last_open_time = now
        while self.blinks and now - self.blinks[0] > 60.0:
            self.blinks.popleft()

    def _update_yawn(self, now, metrics):
        mar = metrics.get("mar")
        active = mar is not None and mar >= float(self.config["mar_threshold"])
        if active:
            if self.yawn_since is None:
                self.yawn_since = now
            self.possible_yawn = (now - self.yawn_since) >= float(self.config["yawn_min_seconds"])
            if self.possible_yawn and (not self.yawns or now - self.yawns[-1] > 2.0):
                self.yawns.append(now)
        else:
            self.yawn_since = None
            self.possible_yawn = False
        while self.yawns and now - self.yawns[0] > 180.0:
            self.yawns.popleft()

    def _update_calibrated_profile(self, now, metrics):
        profile = metrics.get("calibrated_profile", "NO_CALIBRADO")
        confidence = float(metrics.get("calibrated_profile_confidence", 0.0))
        if profile not in ("OPEN", "DROWSY", "ASLEEP") or confidence < self.profile_min_confidence:
            self.calibrated_profile = profile if profile in ("NO_CALIBRADO", "DESCONOCIDO") else "DESCONOCIDO"
            self.calibrated_profile_confidence = confidence
            self.calibrated_profile_since = None
            self.calibrated_profile_seconds = 0.0
            return
        if profile != self.calibrated_profile or self.calibrated_profile_since is None:
            self.calibrated_profile_since = now
        self.calibrated_profile = profile
        self.calibrated_profile_confidence = confidence
        self.calibrated_profile_seconds = max(0.0, now - self.calibrated_profile_since)

    def _update_head(self, now, metrics):
        pitch = metrics.get("pitch")
        active = pitch is not None and abs(float(pitch)) >= float(self.config["head_nod_pitch_threshold"])
        if active:
            if self.nod_since is None:
                self.nod_since = now
            self.possible_nod = (now - self.nod_since) >= float(self.config["head_nod_min_seconds"])
        else:
            self.nod_since = None
            self.possible_nod = False

    def _update_gaze(self, now, metrics):
        gaze = metrics.get("gaze", "DESCONOCIDA")
        away = gaze in ("IZQUIERDA", "DERECHA", "ABAJO")
        if away:
            if self.gaze_away_since is None:
                self.gaze_away_since = now
            self.gaze_away_seconds = now - self.gaze_away_since
        else:
            self.gaze_away_since = None
            self.gaze_away_seconds = 0.0

    def _decide(self, now, m, closed, reliable):
        perclos = m["perclos"]
        cd = self.closed_duration
        blink_display_seconds = min(float(self.config["blink_max_seconds"]), float(self.config["prealert_closed_seconds"]))
        if reliable and closed and cd < blink_display_seconds:
            return self._set("PARPADEO", "Parpadeo en curso", now)
        if cd >= float(self.config["critical_closed_seconds"]):
            return self._set("ALERTA_CRITICA", "Cierre ocular critico: %.1f s" % cd, now)
        if cd >= float(self.config["alert_closed_seconds"]):
            return self._set("ALERTA", "Cierre ocular prolongado: %.1f s" % cd, now)
        if cd >= float(self.config["prealert_closed_seconds"]):
            return self._set("POSIBLE_SOMNOLENCIA", "Cierre ocular sostenido: %.1f s" % cd, now)
        if self.calibrated_profile == "ASLEEP":
            if self.calibrated_profile_seconds >= float(self.config["critical_closed_seconds"]):
                return self._set("ALERTA_CRITICA", "Perfil dormido sostenido: %.1f s" % self.calibrated_profile_seconds, now)
            if self.calibrated_profile_seconds >= float(self.config["alert_closed_seconds"]):
                return self._set("ALERTA", "Perfil dormido: %.1f s" % self.calibrated_profile_seconds, now)
            if self.calibrated_profile_seconds >= float(self.config["prealert_closed_seconds"]):
                return self._set("POSIBLE_SOMNOLENCIA", "Transicion a perfil dormido", now)
        if self.calibrated_profile == "DROWSY" and self.calibrated_profile_seconds >= self.profile_warning_seconds:
            return self._set("POSIBLE_SOMNOLENCIA", "Perfil calibrado de somnolencia: %.1f s" % self.calibrated_profile_seconds, now)
        if perclos >= float(self.config["perclos_alert_threshold"]):
            return self._set("ALERTA", "PERCLOS elevado: %.0f%%" % (perclos * 100.0), now)
        if perclos >= float(self.config["perclos_warning_threshold"]):
            return self._set("POSIBLE_SOMNOLENCIA", "PERCLOS preventivo: %.0f%%" % (perclos * 100.0), now)
        if self.possible_yawn and (cd > 0.3 or len(self.yawns) >= 2):
            return self._set("POSIBLE_SOMNOLENCIA", "Bostezo con evidencia secundaria", now)
        if self.possible_nod and cd > 0.25:
            return self._set("POSIBLE_SOMNOLENCIA", "Cabeceo con cierre ocular", now)
        if self.gaze_away_seconds >= float(self.config["gaze_away_warning_seconds"]):
            return self._set("POSIBLE_SOMNOLENCIA", "Mirada fuera del frente por %.1f s" % self.gaze_away_seconds, now)
        if self.state in ("ALERTA", "ALERTA_CRITICA", "POSIBLE_SOMNOLENCIA") and now - self.last_open_time < float(self.config["recovery_seconds"]):
            return self._set(self.state, "Periodo de recuperacion", now)
        return self._set("NORMAL", "Indicadores dentro de rango", now)

    def _set(self, state, reason, now):
        if state != self.state:
            self.last_transition = now
        self.state = state
        self.reason = reason
        return state, reason
