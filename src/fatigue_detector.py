import time

from .temporal_events import EyeNormalizer, TemporalEventEngine
from .vision_reliability import VisionReliabilityMonitor


class FatigueDetector(object):
    """Coordina señales, eventos y la máquina temporal de somnolencia."""

    STATES = (
        "CALIBRACION", "ALERTA", "SOSPECHA", "SOMNOLENCIA", "CRITICO",
        "RECUPERACION", "MANTENIMIENTO", "PARO_EMERGENCIA", "ERROR",
    )
    OPERATIONAL_STATES = (
        "ALERTA", "SOSPECHA", "SOMNOLENCIA", "CRITICO", "RECUPERACION",
    )
    LEGACY_STATE_ALIASES = {
        "NORMAL": "ALERTA",
        "PARPADEO": "ALERTA",
        "POSIBLE_SOMNOLENCIA": "SOSPECHA",
        "ALERTA_CRITICA": "CRITICO",
    }
    RECOVERY_STATES = ("SOSPECHA", "SOMNOLENCIA", "CRITICO", "RECUPERACION")

    def __init__(self, config, clock=None):
        self.full_config = config
        self.config = config.get("fatigue", {})
        self.stability_config = config.get("stability", {})
        self.clock = clock or time.monotonic
        self.normalizer = EyeNormalizer(config)
        self.vision = VisionReliabilityMonitor(config, clock=self.clock)
        self.events = TemporalEventEngine(config, clock=self.clock)

        self.state = "CALIBRACION"
        self.previous_state = None
        self.reason = "Se requiere un perfil de calibracion valido"
        self.last_transition_reason = self.reason
        self.last_transition = self.clock()
        self.state_entered_at = self.last_transition
        self.stable_since = None
        self.strong_since = None
        self.recovery_open_since = None
        self.last_critical_at = None
        self.operational_state_before_mode = "ALERTA"
        self.profile = None

        # Propiedades heredadas conservadas para UI y configuraciones V3 previas.
        self.ear_threshold = float(self.config.get("ear_threshold", 0.22))
        self.drowsy_ear_threshold = float(
            self.config.get("drowsy_ear_threshold", self.ear_threshold + 0.04)
        )
        self.ear_hysteresis = float(
            self.stability_config.get("ear_hysteresis", 0.012)
        )
        self.ear_open_threshold = self.ear_threshold + self.ear_hysteresis
        self.head_pitch_baseline = None
        self.closed_duration = 0.0
        self.last_blink_duration = 0.0
        self.possible_yawn = False
        self.possible_nod = False
        self.gaze_away_seconds = 0.0
        self.calibrated_profile = "NO_CALIBRADO"
        self.calibrated_profile_confidence = 0.0
        self.calibrated_profile_seconds = 0.0
        self._profile_since = None

    @classmethod
    def canonical_state(cls, state):
        return cls.LEGACY_STATE_ALIASES.get(state, state)

    def reset(self, clear_history=True):
        now = self.clock()
        self.events.reset(clear_history=clear_history)
        self.vision.reset()
        self.state = "ALERTA" if self.profile else "CALIBRACION"
        self.previous_state = None
        self.reason = ("Metricas temporales reiniciadas" if self.profile
                       else "Se requiere un perfil de calibracion valido")
        self.last_transition_reason = self.reason
        self.last_transition = now
        self.state_entered_at = now
        self.stable_since = None
        self.strong_since = None
        self.recovery_open_since = None
        self.closed_duration = 0.0
        self.last_blink_duration = 0.0
        self.possible_yawn = False
        self.possible_nod = False
        self.calibrated_profile = "NO_CALIBRADO"
        self.calibrated_profile_confidence = 0.0
        self.calibrated_profile_seconds = 0.0
        self._profile_since = None

    def apply_calibrated_threshold(self, value):
        self.ear_threshold = float(value)
        self.ear_open_threshold = self.ear_threshold + self.ear_hysteresis

    def apply_calibration(self, result):
        if not result:
            return False
        thresholds = result.get("thresholds", {})
        profile = result.get("profile_data")
        if profile is None and result.get("format_version"):
            profile = result
            thresholds = profile.get("thresholds", thresholds)
        if thresholds.get("ear_threshold") is not None:
            self.ear_threshold = float(thresholds["ear_threshold"])
        elif result.get("ear_threshold") is not None:
            self.ear_threshold = float(result["ear_threshold"])
        if thresholds.get("drowsy_ear_threshold") is not None:
            self.drowsy_ear_threshold = float(thresholds["drowsy_ear_threshold"])
        elif thresholds.get("reduced_ear_threshold") is not None:
            self.drowsy_ear_threshold = float(thresholds["reduced_ear_threshold"])
        if thresholds.get("ear_open_threshold") is not None:
            self.ear_open_threshold = float(thresholds["ear_open_threshold"])
        else:
            self.ear_open_threshold = self.ear_threshold + self.ear_hysteresis
        if thresholds.get("open_pitch_reference") is not None:
            self.head_pitch_baseline = float(thresholds["open_pitch_reference"])
        for key in ("mouth_closed_threshold", "mouth_open_threshold",
                    "mouth_wide_threshold"):
            if thresholds.get(key) is not None:
                self.config[key] = float(thresholds[key])
        if profile is None:
            return False
        self.profile = profile
        self.normalizer.apply_profile(profile)
        self.events.apply_profile(profile)
        neutral = profile.get("neutral_head_pose", {})
        if neutral.get("pitch") is not None:
            self.head_pitch_baseline = float(neutral["pitch"])
        self.reset(clear_history=True)
        return True

    def update(self, metrics, mode="AUTOMATIC", monitoring_enabled=True,
               paused=False, analysis_fps=0.0, camera_error=None):
        now = self.clock()
        self.previous_state = self.state
        vision = self.vision.update(metrics, analysis_fps, camera_error)
        metrics.update({
            "vision_state": vision["state"],
            "vision_reason": vision["reason"],
            "vision_valid_eye_count": vision["valid_eye_count"],
            "vision_measurement_partial": vision["measurement_partial"],
            "no_face_seconds": vision["face_missing_seconds"],
        })

        if mode == "EMERGENCY":
            self.events.update(metrics, monitoring=False, paused=True)
            self.operational_state_before_mode = (
                self.state if self.state in self.OPERATIONAL_STATES else "ALERTA"
            )
            return self._set("PARO_EMERGENCIA", "Paro de emergencia activo", now)
        if mode == "MAINTENANCE":
            self.events.update(metrics, monitoring=False, paused=True)
            self.operational_state_before_mode = (
                self.state if self.state in self.OPERATIONAL_STATES else "ALERTA"
            )
            return self._set(
                "MANTENIMIENTO",
                "Modo mantenimiento: medicion temporal en pausa",
                now,
            )
        if self.state in ("MANTENIMIENTO", "PARO_EMERGENCIA"):
            self.state = self.operational_state_before_mode
            self.state_entered_at = now

        if not monitoring_enabled or self.profile is None:
            event_metrics = self.events.update(metrics, monitoring=False, paused=True)
            metrics.update(event_metrics)
            self._update_compatibility_metrics(metrics)
            return self._set(
                "CALIBRACION",
                "Calibracion estatica y basal de parpadeos incompletas",
                now,
            )
        if paused:
            event_metrics = self.events.update(metrics, monitoring=True, paused=True)
            metrics.update(event_metrics)
            self._update_compatibility_metrics(metrics)
            return self._set(self.state, "Monitoreo pausado; metricas excluidas", now)

        signals = self.normalizer.normalize(metrics)
        signals.update(self.normalizer.pose_deltas(metrics))
        signals.update({
            "face_detected": metrics.get("face_detected", False),
            "pose_reliable": metrics.get("pose_reliable", True),
            "vision_state": vision["state"],
            "frame_sequence": metrics.get("frame_sequence"),
            "ear": metrics.get("ear"),
            "mar": metrics.get("mar"),
        })
        metrics.update(signals)
        event_metrics = self.events.update(signals, monitoring=True, paused=False)
        metrics.update(event_metrics)
        self._update_calibrated_profile(now, metrics)
        self._update_compatibility_metrics(metrics)
        return self._decide(now, metrics)

    def update_no_frame(self, analysis_fps=0.0, camera_error=None,
                        monitoring_enabled=True):
        metrics = {"frame_available": False, "face_detected": False,
                   "quality": 0.0}
        return self.update(metrics, monitoring_enabled=monitoring_enabled,
                           analysis_fps=analysis_fps, camera_error=camera_error)

    def _decide(self, now, metrics):
        vision_state = metrics.get("vision_state")
        valid_eyes = int(metrics.get("valid_eye_count", 0))
        critical = list(metrics.get("critical_evidence", []))
        strong = list(metrics.get("strong_evidence", []))
        weak = list(metrics.get("weak_evidence", []))
        weak_modalities = set(metrics.get("weak_evidence_modalities", []))

        if valid_eyes < 2 and not bool(
            self.config.get("single_eye_critical_enabled", False)
        ):
            critical = []
        relapse_seconds = float(
            self.config.get("recovery_relapse_closed_seconds", 0.8)
        )
        if (self.state == "RECUPERACION"
                and metrics.get("deep_closure_active")
                and metrics.get("current_closure_seconds", 0.0) >= relapse_seconds):
            critical.append("recaida_durante_recuperacion")

        if critical:
            self.last_critical_at = now
            self.stable_since = None
            self.strong_since = None
            self.recovery_open_since = None
            return self._transition(
                "CRITICO",
                "Posible perdida momentanea de vigilancia: %s" %
                self._label(critical[0]),
                now,
            )

        eye_decision_available = bool(metrics.get("eye_measurement_valid", False))
        vision_allows_decision = vision_state in ("VISION_VALIDA", "VISION_DEGRADADA")
        if not eye_decision_available or not vision_allows_decision:
            self.stable_since = None
            self.strong_since = None
            return self._set(
                self.state,
                "Estado conservado por %s: %s" % (
                    vision_state, metrics.get("vision_reason", "medicion no valida")
                ),
                now,
            )

        head_normal = not any(name in metrics.get("active_events", []) for name in (
            "CABEZA_ABAJO", "INCLINACION_CABEZA_SOSTENIDA", "GIRO_LATERAL_CABEZA"
        ))
        eyes_open = (
            metrics.get("closure_normalized") is not None
            and float(metrics["closure_normalized"])
            <= float(self.config.get("recovery_open_closure_level", 0.25))
        )
        stable = not strong and not weak and eyes_open and head_normal

        if self.state == "CRITICO":
            minimum_hold = float(self.config.get("critical_minimum_hold_seconds", 2.0))
            open_required = float(self.config.get("critical_open_recovery_seconds", 2.0))
            if vision_state == "VISION_VALIDA" and eyes_open and head_normal:
                if self.recovery_open_since is None:
                    self.recovery_open_since = now
            else:
                self.recovery_open_since = None
            open_elapsed = (now - self.recovery_open_since
                            if self.recovery_open_since is not None else 0.0)
            if now - self.state_entered_at >= minimum_hold and open_elapsed >= open_required:
                self.stable_since = now
                return self._transition(
                    "RECUPERACION",
                    "Ojos abiertos y cabeza neutral; inicia observacion posterior",
                    now,
                )
            return self._set(
                "CRITICO",
                "Estado critico retenido; requiere vision valida y apertura estable",
                now,
            )

        if self.state == "RECUPERACION":
            if strong:
                self.stable_since = None
                return self._transition(
                    "SOMNOLENCIA",
                    "Patron fuerte durante recuperacion: %s" % self._label(strong[0]),
                    now,
                )
            if stable:
                if self.stable_since is None:
                    self.stable_since = now
                if now - self.stable_since >= float(
                    self.config.get("recovery_observation_seconds", 8.0)
                ):
                    return self._transition(
                        "SOSPECHA",
                        "Recuperacion estable; descenso controlado a sospecha",
                        now,
                    )
            else:
                self.stable_since = None
            return self._set(
                "RECUPERACION",
                "Observando estabilidad posterior al evento critico",
                now,
            )

        if self.state == "SOMNOLENCIA":
            if strong:
                self.stable_since = None
                return self._set(
                    "SOMNOLENCIA", "Evidencia fuerte persistente: %s" %
                    self._label(strong[0]), now,
                )
            if stable:
                if self.stable_since is None:
                    self.stable_since = now
                if now - self.stable_since >= float(
                    self.config.get("somnolence_clear_seconds", 10.0)
                ):
                    return self._transition(
                        "SOSPECHA", "Patron fuerte ausente durante periodo estable", now,
                    )
            else:
                self.stable_since = None
            return self._set("SOMNOLENCIA", "Historial de somnolencia retenido", now)

        if self.state == "SOSPECHA":
            if len(strong) >= 2:
                return self._transition(
                    "SOMNOLENCIA", "Varias evidencias fuertes: %s" %
                    ", ".join(self._label(item) for item in strong[:2]), now,
                )
            if strong:
                self.stable_since = None
                if self.strong_since is None:
                    self.strong_since = now
                if now - self.strong_since >= float(
                    self.config.get("suspicion_to_somnolence_seconds", 4.0)
                ):
                    return self._transition(
                        "SOMNOLENCIA", "Evidencia fuerte persistente: %s" %
                        self._label(strong[0]), now,
                    )
                return self._set(
                    "SOSPECHA", "Confirmando evidencia fuerte: %s" %
                    self._label(strong[0]), now,
                )
            self.strong_since = None
            if len(weak_modalities) >= 2:
                self.stable_since = None
                return self._set(
                    "SOSPECHA", "Evidencias debiles multimodales: %s" %
                    ", ".join(self._label(item) for item in weak), now,
                )
            if stable:
                if self.stable_since is None:
                    self.stable_since = now
                if now - self.stable_since >= float(
                    self.config.get("suspicion_clear_seconds", 8.0)
                ):
                    return self._transition(
                        "ALERTA", "Estabilidad sostenida sin evidencia anomala", now,
                    )
            else:
                self.stable_since = None
            return self._set("SOSPECHA", "Memoria preventiva e histeresis activas", now)

        # ALERTA representa vigilancia normal. Un bostezo aislado no cambia el estado.
        if len(strong) >= 2:
            return self._transition(
                "SOMNOLENCIA", "Combinacion de evidencias fuertes: %s" %
                ", ".join(self._label(item) for item in strong[:2]), now,
            )
        if strong:
            self.strong_since = now
            return self._transition(
                "SOSPECHA", "Evidencia fuerte inicial: %s" % self._label(strong[0]), now,
            )
        if len(weak_modalities) >= 2:
            return self._transition(
                "SOSPECHA", "Dos evidencias debiles de modalidades distintas", now,
            )
        self.stable_since = now if self.stable_since is None else self.stable_since
        return self._set("ALERTA", "Indicadores dentro del comportamiento personal", now)

    @staticmethod
    def _label(name):
        return str(name).replace("_", " ")

    def _update_calibrated_profile(self, now, metrics):
        closure = metrics.get("closure_normalized")
        if closure is None:
            profile = "DESCONOCIDO"
            confidence = 0.0
        else:
            partial = float(metrics.get("partial_closure_reference", 0.5))
            refs = {"OPEN": 0.0, "REDUCED": partial, "CLOSED": 1.0}
            distances = dict((key, abs(float(closure) - value))
                             for key, value in refs.items())
            profile = min(distances, key=distances.get)
            confidence = max(0.0, 1.0 - distances[profile])
        if profile != self.calibrated_profile:
            self._profile_since = now
        self.calibrated_profile = profile
        self.calibrated_profile_confidence = confidence
        self.calibrated_profile_seconds = (
            max(0.0, now - self._profile_since) if self._profile_since is not None else 0.0
        )
        metrics.update({
            "calibrated_profile": profile,
            "calibrated_profile_confidence": confidence,
            "calibrated_profile_seconds": self.calibrated_profile_seconds,
        })

    def _update_compatibility_metrics(self, metrics):
        self.closed_duration = float(metrics.get("current_closure_seconds", 0.0))
        self.last_blink_duration = float(metrics.get("last_blink_seconds", 0.0))
        self.possible_yawn = bool(
            "BOCA_AMPLIAMENTE_ABIERTA" in metrics.get("active_events", [])
        )
        self.possible_nod = bool(
            "CABEZA_ABAJO" in metrics.get("active_events", [])
        )
        metrics.update({
            "closed_seconds": self.closed_duration,
            "possible_yawn": self.possible_yawn,
            "possible_nod": self.possible_nod,
            "ear_threshold": self.ear_threshold,
            "ear_open_threshold": self.ear_open_threshold,
            "drowsy_ear_threshold": self.drowsy_ear_threshold,
        })

    def _transition(self, state, reason, now):
        state = self.canonical_state(state)
        if state != self.state:
            self.last_transition = now
            self.state_entered_at = now
            self.last_transition_reason = reason
            self.stable_since = None
            self.strong_since = None
            self.recovery_open_since = None
        self.state = state
        self.reason = reason
        return state, reason

    def _set(self, state, reason, now):
        state = self.canonical_state(state)
        if state != self.state:
            return self._transition(state, reason, now)
        self.state = state
        self.reason = reason
        return state, reason

    def _set_risk(self, state, reason, now):
        return self._transition(self.canonical_state(state), reason, now)

    # Adaptadores de la API incremental V3 para herramientas de rendimiento.
    def _update_perclos(self, now, closed):
        self.events.perclos_window.update(now, True, True, bool(closed))

    def current_perclos(self, now):
        return self.events.perclos_window.snapshot(now, 0.0, 0.0)["perclos"]

    def _pause_perclos(self, now):
        self.events.perclos_window.pause(now)
