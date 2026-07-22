import time
from collections import deque

class FatigueDetector(object):
    STATES = (
        "INICIALIZANDO", "CALIBRANDO", "NORMAL", "PARPADEO", "POSIBLE_SOMNOLENCIA",
        "ALERTA", "ALERTA_CRITICA", "ROSTRO_NO_DETECTADO", "MANTENIMIENTO",
        "PARO_EMERGENCIA", "ERROR"
    )
    RECOVERY_STATES = ("POSIBLE_SOMNOLENCIA", "ALERTA", "ALERTA_CRITICA")

    def __init__(self, config, clock=None):
        self.config = config["fatigue"]
        self.calibration_config = config.get("calibration", {})
        self.stability_config = config.get("stability", {})
        self.clock = clock or time.monotonic
        self.state = "INICIALIZANDO"
        self.previous_state = None
        self.reason = "Inicializando sistema"
        self.ear_threshold = float(self.config["ear_threshold"])
        self.drowsy_ear_threshold = float(self.config.get("drowsy_ear_threshold", self.ear_threshold + 0.04))
        self.ear_hysteresis = max(
            0.002, float(self.stability_config.get("ear_hysteresis", 0.012))
        )
        self.ear_open_threshold = self.ear_threshold + self.ear_hysteresis
        ear_window = max(1, int(self.stability_config.get("ear_median_window", 3)))
        if ear_window % 2 == 0:
            ear_window += 1
        self.ear_samples = deque(maxlen=ear_window)
        self.eye_quality_threshold = float(
            self.stability_config.get("eye_quality_threshold", 0.25)
        )
        self.close_confirm_seconds = max(
            0.0, float(self.stability_config.get("close_confirm_seconds", 0.08))
        )
        self.open_confirm_seconds = max(
            0.0, float(self.stability_config.get("open_confirm_seconds", 0.15))
        )
        self.unreliable_hold_seconds = max(
            0.0, float(self.stability_config.get("unreliable_hold_seconds", 0.25))
        )
        self.face_loss_hold_seconds = max(
            0.0, float(self.stability_config.get("face_loss_hold_seconds", 0.25))
        )
        self.profile_min_confidence = float(self.calibration_config.get("profile_min_confidence", 0.35))
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
        self.recovery_since = None
        self.last_transition = self.clock()
        self.perclos = deque()
        self._perclos_total_seconds = 0.0
        self._perclos_closed_seconds = 0.0
        self._perclos_last_time = None
        self._perclos_last_closed = False
        self.blinks = deque(maxlen=120)
        self.yawns = deque(maxlen=20)
        self.closed_duration = 0.0
        self.last_blink_duration = 0.0
        self.possible_yawn = False
        self.possible_nod = False
        self.gaze_away_seconds = 0.0
        self.head_pitch_baseline = None
        self._reset_eye_signal(clear_history=True)

    def reset(self):
        threshold = self.ear_threshold
        drowsy_threshold = self.drowsy_ear_threshold
        open_threshold = self.ear_open_threshold
        pitch_baseline = self.head_pitch_baseline
        self.state = "NORMAL"
        self.previous_state = None
        self.reason = "Metricas temporales reiniciadas"
        self.closed_since = None
        self.yawn_since = None
        self.nod_since = None
        self.gaze_away_since = None
        self.no_face_since = None
        self.recovery_since = None
        self.last_transition = self.clock()
        self.perclos.clear()
        self._perclos_total_seconds = 0.0
        self._perclos_closed_seconds = 0.0
        self._perclos_last_time = None
        self._perclos_last_closed = False
        self.blinks.clear()
        self.yawns.clear()
        self.closed_duration = 0.0
        self.last_blink_duration = 0.0
        self.possible_yawn = False
        self.possible_nod = False
        self.gaze_away_seconds = 0.0
        self.ear_threshold = threshold
        self.drowsy_ear_threshold = drowsy_threshold
        self.ear_open_threshold = open_threshold
        self.head_pitch_baseline = pitch_baseline
        self._reset_eye_signal(clear_history=True)
        self.calibrated_profile = "NO_CALIBRADO"
        self.calibrated_profile_confidence = 0.0
        self.calibrated_profile_since = None
        self.calibrated_profile_seconds = 0.0

    def apply_calibrated_threshold(self, value):
        self.ear_threshold = float(value)
        self.ear_open_threshold = self.ear_threshold + self.ear_hysteresis

    def apply_calibration(self, result):
        thresholds = result.get("thresholds", {}) if result else {}
        if thresholds.get("ear_threshold") is not None:
            self.ear_threshold = float(thresholds["ear_threshold"])
        elif result and result.get("ear_threshold") is not None:
            self.ear_threshold = float(result["ear_threshold"])
        if thresholds.get("drowsy_ear_threshold") is not None:
            self.drowsy_ear_threshold = float(thresholds["drowsy_ear_threshold"])
        if thresholds.get("ear_open_threshold") is not None:
            self.ear_open_threshold = float(thresholds["ear_open_threshold"])
        else:
            self.ear_open_threshold = self.ear_threshold + self.ear_hysteresis
        if thresholds.get("open_pitch_reference") is not None:
            self.head_pitch_baseline = float(thresholds["open_pitch_reference"])
        self._reset_eye_signal(clear_history=True)

    def update(self, metrics, mode="AUTOMATIC"):
        now = self.clock()
        self.previous_state = self.state
        if mode == "EMERGENCY":
            self._pause_perclos(now)
            return self._set("PARO_EMERGENCIA", "Paro de emergencia activo", now)
        if mode == "MAINTENANCE":
            self._pause_perclos(now)
            return self._set("MANTENIMIENTO", "Modo mantenimiento: alerta auditiva suspendida", now)
        if not metrics.get("face_detected", False):
            self._pause_perclos(now)
            if self.no_face_since is None:
                self.no_face_since = now
            no_face_time = now - self.no_face_since
            metrics["no_face_seconds"] = no_face_time
            if no_face_time <= self.face_loss_hold_seconds:
                return self._set(
                    self.state,
                    "Rostro inestable: conservando estado %.2f s" % no_face_time,
                    now,
                )
            self._reset_eye_signal(clear_history=True)
            self.calibrated_profile = "DESCONOCIDO"
            self.calibrated_profile_confidence = 0.0
            self.calibrated_profile_since = None
            self.calibrated_profile_seconds = 0.0
            if no_face_time >= float(self.config["no_face_warning_seconds"]):
                return self._set("ROSTRO_NO_DETECTADO", "Rostro no detectado por %.1f s" % no_face_time, now)
            return self._set("NORMAL", "Perdida momentanea de rostro", now)
        self.no_face_since = None
        metrics["no_face_seconds"] = 0.0
        ear = metrics.get("ear_raw", metrics.get("ear"))
        quality = float(metrics.get("quality", 0.0))
        sample_reliable = bool(
            ear is not None
            and quality >= self.eye_quality_threshold
            and metrics.get("eye_reliable", True)
        )
        filtered_ear = self._filter_ear(ear, sample_reliable)
        closed_state, reliable = self._update_eye_state(
            now, filtered_ear, sample_reliable
        )
        closed = bool(closed_state) if closed_state is not None else False
        if closed_state is None:
            self._pause_perclos(now)
        else:
            self._update_perclos(now, closed)
        self._update_eye_timing(now, closed_state, reliable)
        self._update_calibrated_profile(now, metrics)
        self._update_yawn(now, metrics)
        self._update_head(now, metrics)
        self._update_gaze(now, metrics)
        metrics.update({
            "ear_threshold": self.ear_threshold,
            "ear_open_threshold": self.ear_open_threshold,
            "ear_filtered": filtered_ear,
            "eye_closed_stable": closed_state,
            "eye_decision_reliable": reliable,
            "eye_signal_status": self.eye_signal_status,
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

    def _filter_ear(self, ear, reliable):
        if reliable:
            self.ear_samples.append(float(ear))
        if not self.ear_samples:
            return None
        ordered = sorted(self.ear_samples)
        return float(ordered[len(ordered) // 2])

    def _update_eye_state(self, now, ear, sample_reliable):
        self.eye_state_changed = False
        if not sample_reliable or ear is None:
            if (
                self.last_reliable_eye_time is not None
                and now - self.last_reliable_eye_time <= self.unreliable_hold_seconds
            ):
                self.eye_signal_status = "RETENIDO"
                return self.eye_closed, True
            self.eye_candidate_state = None
            self.eye_candidate_since = None
            self.eye_signal_status = "NO_CONFIABLE"
            return None, False

        self.last_reliable_eye_time = now
        if self.eye_closed:
            observed_closed = float(ear) < self.ear_open_threshold
        else:
            observed_closed = float(ear) <= self.ear_threshold

        if observed_closed == self.eye_closed:
            self.eye_candidate_state = None
            self.eye_candidate_since = None
            self.eye_signal_status = "CERRADO" if self.eye_closed else "ABIERTO"
            return self.eye_closed, True

        if self.eye_candidate_state != observed_closed or self.eye_candidate_since is None:
            self.eye_candidate_state = observed_closed
            self.eye_candidate_since = now

        confirm_seconds = (
            self.close_confirm_seconds if observed_closed else self.open_confirm_seconds
        )
        if now - self.eye_candidate_since >= confirm_seconds:
            self.eye_closed = observed_closed
            self.eye_state_since = self.eye_candidate_since
            self.eye_state_changed = True
            self.eye_state_transition_at = self.eye_candidate_since
            self.eye_candidate_state = None
            self.eye_candidate_since = None
            self.eye_signal_status = "CERRADO" if self.eye_closed else "ABIERTO"
        else:
            self.eye_signal_status = "CONFIRMANDO_CIERRE" if observed_closed else "CONFIRMANDO_APERTURA"
        return self.eye_closed, True

    def _reset_eye_signal(self, clear_history=False):
        if clear_history and hasattr(self, "ear_samples"):
            self.ear_samples.clear()
        self.eye_closed = False
        self.eye_candidate_state = None
        self.eye_candidate_since = None
        self.eye_state_since = None
        self.eye_state_changed = False
        self.eye_state_transition_at = None
        self.last_reliable_eye_time = None
        self.eye_signal_status = "INICIAL"
        self.closed_since = None
        self.closed_duration = 0.0

    def _update_perclos(self, now, closed):
        if self._perclos_last_time is not None:
            start = self._perclos_last_time
            duration = max(0.0, now - start)
            if duration > 0.0:
                segment = (start, now, self._perclos_last_closed)
                self.perclos.append(segment)
                self._perclos_total_seconds += duration
                if self._perclos_last_closed:
                    self._perclos_closed_seconds += duration
        self._perclos_last_time = now
        self._perclos_last_closed = bool(closed)
        self._prune_perclos(now)

    def current_perclos(self, now):
        self._prune_perclos(now)
        if self._perclos_total_seconds <= 0.0:
            return 0.0
        value = self._perclos_closed_seconds / self._perclos_total_seconds
        return max(0.0, min(1.0, value))

    def _pause_perclos(self, now):
        self._perclos_last_time = None
        self._perclos_last_closed = False
        self._prune_perclos(now)

    def _prune_perclos(self, now):
        cutoff = now - float(self.config["perclos_window_seconds"])
        while self.perclos and self.perclos[0][1] <= cutoff:
            start, end, closed = self.perclos.popleft()
            duration = max(0.0, end - start)
            self._perclos_total_seconds -= duration
            if closed:
                self._perclos_closed_seconds -= duration
        if self.perclos and self.perclos[0][0] < cutoff:
            start, end, closed = self.perclos.popleft()
            removed = max(0.0, cutoff - start)
            self._perclos_total_seconds -= removed
            if closed:
                self._perclos_closed_seconds -= removed
            self.perclos.appendleft((cutoff, end, closed))
        self._perclos_total_seconds = max(0.0, self._perclos_total_seconds)
        self._perclos_closed_seconds = max(
            0.0,
            min(self._perclos_closed_seconds, self._perclos_total_seconds),
        )

    def _update_eye_timing(self, now, closed, reliable):
        if not reliable or closed is None:
            return
        if closed:
            if self.closed_since is None:
                self.closed_since = self.eye_state_since if self.eye_state_since is not None else now
            self.closed_duration = now - self.closed_since
        else:
            if self.eye_state_changed and self.closed_since is not None:
                ended_at = self.eye_state_transition_at if self.eye_state_transition_at is not None else now
                dur = max(0.0, ended_at - self.closed_since)
                if float(self.config["blink_min_seconds"]) <= dur <= float(self.config["blink_max_seconds"]):
                    self.blinks.append(now)
                    self.last_blink_duration = dur
            self.closed_since = None
            self.closed_duration = 0.0
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
        pitch_delta = None
        if pitch is not None:
            reference = self.head_pitch_baseline if self.head_pitch_baseline is not None else 0.0
            pitch_delta = float(pitch) - reference
        metrics["pitch_delta"] = pitch_delta
        active = pitch_delta is not None and abs(pitch_delta) >= float(self.config["head_nod_pitch_threshold"])
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
            return self._set_risk("ALERTA_CRITICA", "Cierre ocular critico: %.1f s" % cd, now)
        if cd >= float(self.config["alert_closed_seconds"]):
            return self._set_risk("ALERTA", "Cierre ocular prolongado: %.1f s" % cd, now)
        if cd >= float(self.config["prealert_closed_seconds"]):
            return self._set_risk("POSIBLE_SOMNOLENCIA", "Cierre ocular sostenido: %.1f s" % cd, now)
        if self.calibrated_profile == "ASLEEP":
            if self.calibrated_profile_seconds >= float(self.config["critical_closed_seconds"]):
                return self._set_risk("ALERTA_CRITICA", "Perfil dormido sostenido: %.1f s" % self.calibrated_profile_seconds, now)
            if self.calibrated_profile_seconds >= float(self.config["alert_closed_seconds"]):
                return self._set_risk("ALERTA", "Perfil dormido: %.1f s" % self.calibrated_profile_seconds, now)
            if self.calibrated_profile_seconds >= float(self.config["prealert_closed_seconds"]):
                return self._set_risk("POSIBLE_SOMNOLENCIA", "Transicion a perfil dormido", now)
        if self.calibrated_profile == "DROWSY" and self.calibrated_profile_seconds >= self.profile_warning_seconds:
            return self._set_risk("POSIBLE_SOMNOLENCIA", "Perfil calibrado de somnolencia: %.1f s" % self.calibrated_profile_seconds, now)
        if perclos >= float(self.config["perclos_alert_threshold"]):
            return self._set_risk("ALERTA", "PERCLOS elevado: %.0f%%" % (perclos * 100.0), now)
        if perclos >= float(self.config["perclos_warning_threshold"]):
            return self._set_risk("POSIBLE_SOMNOLENCIA", "PERCLOS preventivo: %.0f%%" % (perclos * 100.0), now)
        if self.possible_yawn and (cd > 0.3 or len(self.yawns) >= 2):
            return self._set_risk("POSIBLE_SOMNOLENCIA", "Bostezo con evidencia secundaria", now)
        if self.possible_nod and cd > 0.25:
            return self._set_risk("POSIBLE_SOMNOLENCIA", "Cabeceo con cierre ocular", now)
        if self.gaze_away_seconds >= float(self.config["gaze_away_warning_seconds"]):
            return self._set_risk("POSIBLE_SOMNOLENCIA", "Mirada fuera del frente por %.1f s" % self.gaze_away_seconds, now)
        if self.state in self.RECOVERY_STATES:
            if self.recovery_since is None:
                self.recovery_since = now
            recovery_elapsed = now - self.recovery_since
            if recovery_elapsed < float(self.config["recovery_seconds"]):
                return self._set(
                    self.state,
                    "Periodo de recuperacion: %.1f s" % recovery_elapsed,
                    now,
                )
        return self._set("NORMAL", "Indicadores dentro de rango", now)

    def _set_risk(self, state, reason, now):
        self.recovery_since = None
        return self._set(state, reason, now)

    def _set(self, state, reason, now):
        if state not in self.RECOVERY_STATES:
            self.recovery_since = None
        if state != self.state:
            self.last_transition = now
        self.state = state
        self.reason = reason
        return state, reason
