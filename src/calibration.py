import json
import os
import time

import numpy as np


class CalibrationManager(object):
    FORMAT_VERSION = 2
    STATIC_PROFILES = ("OPEN", "REDUCED", "CLOSED")
    PROFILES = STATIC_PROFILES
    PROFILE_ALIASES = {
        "OPEN": "OPEN", "O": "OPEN",
        "DROWSY": "REDUCED", "REDUCED": "REDUCED",
        "APERTURA_OCULAR_REDUCIDA": "REDUCED", "S": "REDUCED",
        "ASLEEP": "CLOSED", "CLOSED": "CLOSED", "D": "CLOSED",
    }
    PROFILE_LABELS = {
        "OPEN": "OJOS NORMALMENTE ABIERTOS",
        "REDUCED": "APERTURA OCULAR REDUCIDA",
        "CLOSED": "OJOS COMPLETAMENTE CERRADOS",
        "DYNAMIC_NATURAL": "OBSERVACION NATURAL DE PARPADEOS",
        "DYNAMIC_VOLUNTARY": "PARPADEOS VOLUNTARIOS NORMALES",
        "YAWN": "CALIBRACION PERSONAL DE BOSTEZOS",
    }
    PROFILE_KEYS = {"OPEN": "O", "REDUCED": "R", "CLOSED": "C"}
    INSTRUCTIONS = {
        "OPEN": ("Mire al frente con postura natural; no abra exageradamente "
                 "los ojos y evite mover la cabeza"),
        "REDUCED": ("Mantenga los parpados parcialmente cerrados, sin cerrarlos "
                    "por completo"),
        "CLOSED": ("Cierre ambos ojos y mantengalos cerrados de forma natural "
                   "durante toda la captura"),
        "DYNAMIC_NATURAL": "Mire al frente de manera natural durante un minuto",
        "DYNAMIC_VOLUNTARY": ("Realice entre cinco y ocho parpadeos normales, "
                              "sin exagerarlos"),
        "YAWN": ("Realice bostezos completos: abra ampliamente la boca, "
                 "mantengala abierta y cierrela entre cada intento"),
    }
    FEATURES = ("ear", "left_ear", "right_ear", "mar", "pitch", "yaw", "roll")

    def __init__(self, config, clock=None):
        self.full_config = config
        self.config = config.get("calibration", {})
        self.clock = clock or time.monotonic
        self.duration = float(self.config.get("duration_seconds", 5.0))
        self.preparation_seconds = max(
            0.0, float(self.config.get("preparation_seconds", 0.0))
        )
        self.require_stage_confirmation = bool(
            self.config.get("require_stage_confirmation", False)
        )
        self.min_samples = int(self.config.get("min_samples", 30))
        self.quality_threshold = float(self.config.get("quality_threshold", 0.30))
        stability = config.get("stability", {})
        self.min_eye_width = float(stability.get("min_eye_width_pixels", 10.0))
        self.min_eye_sharpness = float(stability.get("min_eye_sharpness", 12.0))
        self.min_valid_ratio = float(self.config.get("min_valid_sample_ratio", 0.65))
        self.min_ear_gap = float(self.config.get("min_ear_gap", 0.020))
        self.min_open_closed_gap = float(self.config.get("min_open_closed_gap", 0.060))
        self.max_ear_std = float(self.config.get("max_ear_std", 0.035))
        self.max_ear_mad = float(self.config.get("max_ear_mad", 0.025))
        self.max_reduced_ear_std = float(
            self.config.get("max_reduced_ear_std", 0.055)
        )
        self.max_reduced_ear_mad = float(
            self.config.get("max_reduced_ear_mad", 0.040)
        )
        self.max_head_std = float(self.config.get("max_head_angle_std_degrees", 6.0))
        self.max_eye_asymmetry_ratio = float(
            self.config.get("max_eye_asymmetry_ratio", 0.35)
        )
        self.max_partial_eye_difference = float(
            self.config.get("max_partial_eye_difference", 0.25)
        )
        self.profile_path = self.config.get("profile_path", "calibration_profile.json")
        self.parameters_path = self.config.get(
            "parameters_path", "calibration_parameters.json"
        )

        self.active = False
        self.active_profile = None
        self.started = 0.0
        self.capture_started = 0.0
        self.preparing = False
        self.awaiting_confirmation = False
        self.pending_confirmation_profile = None
        self.initial_prompt = False
        self.skipped = False
        self.samples = []
        self.sample_attempts = 0
        self.sample_rejections = {}
        self.progress = 0.0
        self.profiles = {}
        self.thresholds = {}
        self.static_ready = False
        self.model_ready = False
        self.pending_recalibration = False
        self.profile_data = None
        self.result_threshold = None
        self.last_result = None
        self.last_failed_stage = None
        self.message = "Calibracion requerida"

        self.dynamic_mode = None
        self.dynamic_state = "OPEN"
        self.dynamic_event = None
        self.dynamic_invalid = False
        self.dynamic_natural_events = []
        self.dynamic_voluntary_events = []
        self.dynamic_closure = None
        self.dynamic_max_closure = 0.0
        self.dynamic_rejected_events = 0
        self.yawn_calibration_events = []
        self.yawn_calibration_state = "NORMAL"
        self.yawn_calibration_event = None
        self.yawn_calibration_mar = None
        self.yawn_calibration_max_mar = 0.0
        self.mouth_calibration = None
        self.yawn_only_recalibration = False
        self._load_profile()

    def ready_for_monitoring(self):
        return bool(self.model_ready and self.profile_data and not self.active
                    and not self.awaiting_confirmation
                    and not self.pending_recalibration)

    def request_initial_calibration(self):
        self.active = False
        self.initial_prompt = True
        self.awaiting_confirmation = True
        self.pending_confirmation_profile = "FULL"
        self.pending_recalibration = True
        self.message = ("No existe una calibracion valida. Presione ESPACIO, "
                        "ENTER o haga click para comenzar; presione N para omitir.")
        return True

    def skip_calibration(self):
        fatigue = self.full_config.get("fatigue", {})
        threshold = float(fatigue.get("ear_threshold", 0.22))
        opened = max(threshold + 0.06, 0.30)
        closed = max(0.05, threshold - 0.08)
        reduced = (opened + closed) * 0.5
        eyes = dict((side, {"open": opened, "reduced": reduced, "closed": closed})
                    for side in ("left", "right", "combined"))
        self.thresholds = {
            "ear_threshold": threshold,
            "ear_open_threshold": threshold + float(
                self.full_config.get("stability", {}).get("ear_hysteresis", 0.012)
            ),
            "reduced_ear_threshold": (opened + reduced) * 0.5,
            "drowsy_ear_threshold": (opened + reduced) * 0.5,
        }
        self.profile_data = {
            "format_version": self.FORMAT_VERSION,
            "eyes": eyes,
            "partial_closure_normalized": 0.5,
            "neutral_head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "thresholds": dict(self.thresholds),
            "stages": {},
            "quality": {"fallback": True},
            "blink_baseline": {"selected": {"source": "fallback", "event_count": 0}},
            "mouth": {
                "baseline_mar": float(fatigue.get("mouth_closed_threshold", 0.38)),
                "closed_threshold": float(fatigue.get("mouth_closed_threshold", 0.38)),
                "open_threshold": float(fatigue.get("mouth_open_threshold", 0.48)),
                "wide_threshold": float(fatigue.get("mouth_wide_threshold", 0.65)),
                "event_count": 0, "fallback": True,
            },
        }
        self.model_ready = True
        self.static_ready = False
        self.skipped = True
        self.initial_prompt = False
        self.active = False
        self.awaiting_confirmation = False
        self.pending_confirmation_profile = None
        self.pending_recalibration = False
        self.message = "Calibracion omitida; monitoreo con parametros de respaldo"
        return self.profile_data

    def next_required_stage(self):
        for profile in self.STATIC_PROFILES:
            if profile not in self.profiles:
                return profile
        return None

    def start(self, profile="OPEN"):
        requested = str(profile or "OPEN").upper()
        self.initial_prompt = False
        self.skipped = False
        self.awaiting_confirmation = False
        self.pending_confirmation_profile = None
        if requested in ("FULL", "SEQUENCE", "CALIBRATION"):
            self.profiles = {}
            self.thresholds = {}
            self.static_ready = False
            self.dynamic_natural_events = []
            self.dynamic_voluntary_events = []
            self.yawn_calibration_events = []
            self.yawn_only_recalibration = False
            self.pending_recalibration = True
            requested = "OPEN"
        if requested in ("DYNAMIC", "DYNAMIC_NATURAL"):
            if not self.static_ready:
                raise ValueError("Complete primero las tres etapas estaticas")
            return self._start_dynamic("NATURAL")
        if requested == "DYNAMIC_VOLUNTARY":
            if not self.static_ready:
                raise ValueError("Complete primero las tres etapas estaticas")
            return self._start_dynamic("VOLUNTARY")
        if requested in ("YAWN", "BOSTEZO", "BOSTEZOS"):
            if not self.static_ready:
                raise ValueError("Complete primero las tres etapas estaticas")
            if not (self.dynamic_natural_events or self.dynamic_voluntary_events
                    or self.profile_data):
                raise ValueError("Complete primero la calibracion de parpadeos")
            return self._start_yawn_calibration()

        profile = self.PROFILE_ALIASES.get(requested)
        if profile not in self.STATIC_PROFILES:
            raise ValueError("Perfil de calibracion desconocido: %s" % requested)
        self.pending_recalibration = True
        self.active = True
        self.active_profile = profile
        self.started = self.clock()
        self.capture_started = self.started + self.preparation_seconds
        self.preparing = self.preparation_seconds > 0.0
        self.samples = []
        self.sample_attempts = 0
        self.sample_rejections = {}
        self.progress = 0.0
        self.last_result = None
        self.last_failed_stage = None
        self.message = self._stage_message(profile)

    def update(self, metrics):
        if not self.active:
            return None
        if self.active_profile in ("DYNAMIC_NATURAL", "DYNAMIC_VOLUNTARY"):
            return self._update_dynamic(metrics)
        if self.active_profile == "YAWN":
            return self._update_yawn_calibration(metrics)

        now = self.clock()
        if self._preparation_active(now):
            return None
        self.sample_attempts += 1
        # La preparacion y la captura son periodos consecutivos. Medir desde
        # ``started`` descontaba la preparacion de la duracion util y, a FPS
        # bajos, hacia imposible reunir ``min_samples``.
        self.progress = min(
            1.0,
            (now - self.capture_started) / max(0.1, self.duration),
        )
        sample = self._extract_sample(metrics, require_quality=True)
        if sample is not None:
            self.samples.append(sample)
        if self.progress < 1.0:
            return None

        profile = self.active_profile
        self.active = False
        self.active_profile = None
        summary, error = self._summarize(self.samples, self.sample_attempts, profile)
        if summary is None:
            self.last_failed_stage = profile
            self.message = "Etapa %s invalida: %s. Repita solo esta etapa." % (
                self.PROFILE_LABELS[profile], error,
            )
            self.last_result = {
                "accepted": False, "profile": profile, "failed_stage": profile,
                "reason": self.message, "model_ready": False,
            }
            return self.last_result

        self.profiles[profile] = summary
        self._rebuild_static_model()
        accepted = bool(profile in self.profiles and self.last_failed_stage is None)
        result = {
            "accepted": accepted, "profile": profile, "summary": summary,
            "profiles": dict(self.profiles), "thresholds": dict(self.thresholds),
            "static_ready": self.static_ready, "model_ready": False,
            "reason": self.message,
        }
        if not accepted and self.last_failed_stage is not None:
            result["failed_stage"] = self.last_failed_stage
        if profile == "OPEN":
            provisional = max(0.08, min(0.34, summary["ear"]["median"] * 0.72))
            result["ear_threshold"] = provisional
            self.result_threshold = provisional
        if self.static_ready:
            if self.require_stage_confirmation:
                result["next_stage"] = "DYNAMIC_NATURAL"
                self.message = self._confirmation_message("DYNAMIC_NATURAL")
                result["reason"] = self.message
            else:
                self._start_dynamic("NATURAL")
                result["dynamic_started"] = True
                result["reason"] = self.message
        self.last_result = result
        return result

    def await_next_stage(self, profile, message=None):
        requested = str(profile or "").upper()
        if requested in ("DYNAMIC", "DYNAMIC_NATURAL"):
            requested = "DYNAMIC_NATURAL"
        elif requested == "DYNAMIC_VOLUNTARY":
            requested = "DYNAMIC_VOLUNTARY"
        elif requested in ("YAWN", "BOSTEZO", "BOSTEZOS"):
            requested = "YAWN"
        else:
            requested = self.PROFILE_ALIASES.get(requested)
        valid_dynamic = requested in ("DYNAMIC_NATURAL", "DYNAMIC_VOLUNTARY", "YAWN")
        if requested not in self.STATIC_PROFILES and not valid_dynamic:
            return False
        self.active = False
        self.active_profile = None
        self.preparing = False
        self.progress = 0.0
        self.samples = []
        self.sample_attempts = 0
        self.sample_rejections = {}
        self.awaiting_confirmation = True
        self.pending_confirmation_profile = requested
        self.message = message or self._confirmation_message(requested)
        return True

    def confirm_pending_stage(self):
        if not self.awaiting_confirmation or not self.pending_confirmation_profile:
            return False
        profile = self.pending_confirmation_profile
        self.initial_prompt = False
        self.awaiting_confirmation = False
        self.pending_confirmation_profile = None
        self.start(profile)
        return True

    def classify(self, metrics):
        if not self.profile_data:
            return {"profile": "NO_CALIBRADO", "confidence": 0.0, "scores": {}}
        closure = self.normalized_closure(metrics)
        if closure is None:
            return {"profile": "DESCONOCIDO", "confidence": 0.0, "scores": {}}
        partial = float(self.profile_data.get("partial_closure_normalized", 0.5))
        references = {"OPEN": 0.0, "REDUCED": partial, "CLOSED": 1.0}
        scores = dict((name, abs(float(closure) - value))
                      for name, value in references.items())
        profile = min(scores, key=scores.get)
        second = sorted(scores.values())[1]
        confidence = max(0.0, min(1.0,
            (second - scores[profile]) / max(second, 1e-6)))
        return {"profile": profile, "confidence": confidence, "scores": scores}

    def normalized_closure(self, metrics):
        if not self.profile_data:
            return None
        closures = []
        fallback_reliable = bool(metrics.get("eye_reliable", False))
        for side, key, reliable_key in (
            ("left", "left_ear", "left_eye_reliable"),
            ("right", "right_ear", "right_eye_reliable"),
        ):
            if not bool(metrics.get(reliable_key, fallback_reliable)):
                continue
            value = metrics.get(key)
            ref = self.profile_data.get("eyes", {}).get(side, {})
            opened, closed = ref.get("open"), ref.get("closed")
            if value is None or opened is None or closed is None:
                continue
            denominator = float(opened) - float(closed)
            if denominator > 1e-6:
                closures.append(max(0.0, min(1.0,
                    (float(opened) - float(value)) / denominator)))
        if closures:
            return sum(closures) / float(len(closures))
        ear = metrics.get("ear")
        combined = self.profile_data.get("eyes", {}).get("combined", {})
        denominator = float(combined.get("open", 0.0)) - float(combined.get("closed", 0.0))
        if ear is not None and denominator > 1e-6:
            return max(0.0, min(1.0,
                (float(combined["open"]) - float(ear)) / denominator))
        return None

    def status_metrics(self):
        status = []
        for profile in self.STATIC_PROFILES:
            status.append("%s:%s" % (self.PROFILE_KEYS[profile],
                "OK" if profile in self.profiles else "--"))
        blink_count = (len(self.dynamic_natural_events) if self.dynamic_mode == "NATURAL"
                       else len(self.dynamic_voluntary_events))
        prepare_remaining = 0.0
        if self.active and self.preparing:
            prepare_remaining = max(0.0, self.capture_started - self.clock())
        return {
            "calibration_active": self.active or self.awaiting_confirmation,
            "calibration_initial_prompt": self.initial_prompt,
            "calibration_skipped": self.skipped,
            "calibration_target": (
                "OPEN" if self.initial_prompt else
                (self.pending_confirmation_profile if self.awaiting_confirmation
                 else self.active_profile or "")
            ),
            "calibration_progress": self.progress if self.active else 0.0,
            "calibration_preparing": bool(self.active and self.preparing),
            "calibration_waiting_confirmation": bool(self.awaiting_confirmation),
            "calibration_prepare_remaining_seconds": prepare_remaining,
            "calibration_ready": self.ready_for_monitoring(),
            "calibration_static_ready": self.static_ready,
            "calibration_profiles": " ".join(status),
            "calibration_message": self.message,
            "calibration_failed_stage": self.last_failed_stage or "",
            "calibration_valid_samples": (len(self.samples)
                if self.active_profile in self.STATIC_PROFILES else 0),
            "calibration_sample_attempts": self.sample_attempts,
            "calibration_rejections": dict(self.sample_rejections),
            "calibration_rejection_reason": (max(
                self.sample_rejections, key=self.sample_rejections.get
            ) if self.sample_rejections else ""),
            "calibration_blink_count": blink_count,
            "calibration_dynamic_state": self.dynamic_state,
            "calibration_dynamic_closure": self.dynamic_closure,
            "calibration_dynamic_max_closure": self.dynamic_max_closure,
            "calibration_natural_blinks": len(self.dynamic_natural_events),
            "calibration_voluntary_blinks": len(self.dynamic_voluntary_events),
            "calibration_yawn_count": len(self.yawn_calibration_events),
            "calibration_yawn_state": self.yawn_calibration_state,
            "calibration_yawn_mar": self.yawn_calibration_mar,
            "calibration_yawn_max_mar": self.yawn_calibration_max_mar,
            "calibrated_ear_threshold": self.thresholds.get("ear_threshold"),
            "calibrated_ear_open_threshold": self.thresholds.get("ear_open_threshold"),
            "reduced_ear_threshold": self.thresholds.get("reduced_ear_threshold"),
            "calibration_profile_path": self.profile_path,
            "calibration_parameters_path": self.parameters_path,
            "calibration_format_version": (self.profile_data.get("format_version")
                if self.profile_data else None),
        }

    def _reject_sample(self, reason):
        self.sample_rejections[reason] = self.sample_rejections.get(reason, 0) + 1
        return None

    def _extract_sample(self, metrics, require_quality, allow_closed_eyes=False):
        if not metrics.get("face_detected", False):
            return self._reject_sample("rostro no detectado")
        quality = float(metrics.get("quality", 0.0))
        if require_quality and quality < self.quality_threshold:
            return self._reject_sample("calidad facial baja")
        # La pose absoluta depende de donde pueda instalarse la camara en cada
        # vehiculo. La calibracion aprende esa pose neutral y _summarize exige
        # que sea estable; no reutiliza el limite frontal de supervision.
        fallback_eye = metrics.get("ear")
        left_ear = metrics.get("left_ear", fallback_eye)
        right_ear = metrics.get("right_ear", fallback_eye)
        raw_eye_quality = all(key in metrics for key in (
            "left_eye_width", "right_eye_width",
            "left_eye_sharpness", "right_eye_sharpness",
        ))
        if raw_eye_quality and not allow_closed_eyes:
            if float(metrics["left_eye_width"]) < self.min_eye_width:
                return self._reject_sample("ojo izquierdo demasiado pequeno")
            if float(metrics["right_eye_width"]) < self.min_eye_width:
                return self._reject_sample("ojo derecho demasiado pequeno")
            if float(metrics["left_eye_sharpness"]) < self.min_eye_sharpness:
                return self._reject_sample("ojo izquierdo desenfocado")
            if float(metrics["right_eye_sharpness"]) < self.min_eye_sharpness:
                return self._reject_sample("ojo derecho desenfocado")
        else:
            fallback_reliable = bool(metrics.get("eye_reliable", True))
            if not bool(metrics.get("left_eye_reliable", fallback_reliable)):
                return self._reject_sample("ojo izquierdo no confiable")
            if not bool(metrics.get("right_eye_reliable", fallback_reliable)):
                return self._reject_sample("ojo derecho no confiable")
        if not self._valid_ear(left_ear) or not self._valid_ear(right_ear):
            return self._reject_sample("EAR ocular ausente o fuera de rango")
        left_ear, right_ear = float(left_ear), float(right_ear)
        sample = {
            "ear": (left_ear + right_ear) * 0.5,
            "left_ear": left_ear, "right_ear": right_ear, "quality": quality,
        }
        for feature in ("mar", "pitch", "yaw", "roll"):
            value = metrics.get(feature)
            if value is not None and np.isfinite(float(value)):
                sample[feature] = float(value)
        return sample

    def _valid_ear(self, value):
        if value is None:
            return False
        value = float(value)
        return bool(np.isfinite(value)
            and value >= float(self.config.get("min_possible_ear", 0.03))
            and value <= float(self.config.get("max_possible_ear", 0.60)))

    def _summarize(self, samples, attempts, profile=None):
        if len(samples) < self.min_samples:
            detail = ""
            if self.sample_rejections:
                ordered = sorted(
                    self.sample_rejections.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                detail = "; rechazos: " + ", ".join(
                    "%s=%d" % item for item in ordered
                )
            return None, "solo %d muestras validas; se requieren %d%s" % (
                len(samples), self.min_samples, detail)
        valid_ratio = len(samples) / float(max(1, attempts))
        if valid_ratio < self.min_valid_ratio:
            return None, "cobertura valida %.0f%% inferior al minimo %.0f%%" % (
                valid_ratio * 100.0, self.min_valid_ratio * 100.0)
        summary = {"sample_count": len(samples), "attempted_samples": int(attempts),
                   "valid_sample_ratio": valid_ratio}
        stats_samples = samples
        if len(samples) > 10:
            ordered = sorted(samples, key=lambda item: item["ear"])
            if profile == "CLOSED":
                keep = max(10, int(round(len(samples) * 0.60)))
                stats_samples = ordered[:keep]
            elif profile == "OPEN":
                trim = max(1, int(round(len(samples) * 0.10)))
                stats_samples = ordered[trim:-trim]
            elif profile == "REDUCED":
                trim = max(1, int(round(len(samples) * 0.20)))
                stats_samples = ordered[trim:-trim]
            summary["robust_sample_count"] = len(stats_samples)
        for feature in self.FEATURES:
            values = [sample[feature] for sample in stats_samples if feature in sample]
            if values:
                summary[feature] = self._statistics(values)
        if "ear" not in summary:
            return None, "EAR no disponible"
        ear_std_limit = (self.max_reduced_ear_std
            if profile == "REDUCED" else self.max_ear_std)
        ear_mad_limit = (self.max_reduced_ear_mad
            if profile == "REDUCED" else self.max_ear_mad)
        for feature in ("ear", "left_ear", "right_ear"):
            stats = summary.get(feature)
            if stats is None:
                return None, "faltan mediciones de ambos ojos"
            if (profile != "REDUCED"
                    and (stats["robust_std"] > ear_std_limit
                         or stats["mad"] > ear_mad_limit)):
                return None, "variacion ocular excesiva en %s" % feature
        for feature in ("pitch", "yaw", "roll"):
            stats = summary.get(feature)
            if stats is not None and stats["robust_std"] > self.max_head_std:
                return None, "movimiento excesivo de cabeza en %s" % feature
        left, right = summary["left_ear"]["median"], summary["right_ear"]["median"]
        asymmetry = abs(left - right) / max(1e-6, (left + right) * 0.5)
        if asymmetry > self.max_eye_asymmetry_ratio:
            return None, "diferencia entre ojos %.0f%% superior al maximo" % (
                asymmetry * 100.0)
        summary["eye_asymmetry_ratio"] = asymmetry
        summary["quality"] = {
            "median": float(np.median([sample["quality"] for sample in stats_samples])),
            "valid_sample_ratio": valid_ratio, "eye_asymmetry_ratio": asymmetry,
        }
        return summary, None

    @staticmethod
    def _statistics(values):
        arr = np.asarray(values, dtype=float)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        return {"median": median, "std": float(np.std(arr)), "mad": mad,
                "robust_std": 1.4826 * mad,
                "p10": float(np.percentile(arr, 10)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90))}

    def _rebuild_static_model(self):
        missing = [self.PROFILE_LABELS[p] for p in self.STATIC_PROFILES
                   if p not in self.profiles]
        if missing:
            self.static_ready = False
            self.last_failed_stage = None
            self.thresholds = {}
            self.message = "Etapa guardada. Falta calibrar: %s" % ", ".join(missing)
            return
        features = ("ear", "left_ear", "right_ear")
        anchors_valid = all(
            self.profiles["OPEN"][feature]["median"]
            - self.profiles["CLOSED"][feature]["median"]
            >= self.min_open_closed_gap
            for feature in features
        )
        if not anchors_valid:
            values = " ".join(
                "%s=%.3f>%.3f" % (
                    feature,
                    self.profiles["OPEN"][feature]["median"],
                    self.profiles["CLOSED"][feature]["median"],
                )
                for feature in features
            )
            self.static_ready = False
            self.thresholds = {}
            self.last_failed_stage = "CLOSED"
            self.profiles.pop("CLOSED", None)
            self.message = ("Ojos abiertos y cerrados no son distinguibles "
                            "(%s). Repita OJOS COMPLETAMENTE CERRADOS." % values)
            return

        reduced_adjusted = False
        for feature in features:
            opened = self.profiles["OPEN"][feature]["median"]
            reduced = self.profiles["REDUCED"][feature]["median"]
            closed = self.profiles["CLOSED"][feature]["median"]
            if (opened - reduced < self.min_ear_gap
                    or reduced - closed < self.min_ear_gap):
                # La apertura parcial no es una postura repetible. Cuando los
                # anclajes si son buenos, se obtiene una referencia personal
                # intermedia en vez de encerrar al usuario en reintentos.
                self.profiles["REDUCED"][feature]["median"] = (opened + closed) * 0.5
                reduced_adjusted = True
        partials = []
        for feature in ("left_ear", "right_ear", "ear"):
            opened = self.profiles["OPEN"][feature]["median"]
            reduced = self.profiles["REDUCED"][feature]["median"]
            closed = self.profiles["CLOSED"][feature]["median"]
            partials.append((opened - reduced) / max(1e-6, opened - closed))
        if abs(partials[0] - partials[1]) > self.max_partial_eye_difference:
            self.static_ready = False
            self.thresholds = {}
            self.last_failed_stage = "REDUCED"
            self.profiles.pop("REDUCED", None)
            self.message = ("La apertura reducida no es consistente entre ambos ojos; "
                            "repita solo APERTURA OCULAR REDUCIDA")
            return
        opened = self.profiles["OPEN"]["ear"]["median"]
        reduced = self.profiles["REDUCED"]["ear"]["median"]
        closed = self.profiles["CLOSED"]["ear"]["median"]
        hysteresis = float(self.full_config.get("stability", {}).get("ear_hysteresis", 0.012))
        deep_threshold = (reduced + closed) * 0.5
        self.thresholds = {
            "reduced_ear_threshold": (opened + reduced) * 0.5,
            "drowsy_ear_threshold": (opened + reduced) * 0.5,
            "ear_threshold": deep_threshold,
            "ear_open_threshold": min(reduced, deep_threshold + hysteresis),
            "open_ear_reference": opened,
            "reduced_ear_reference": reduced,
            "drowsy_ear_reference": reduced,
            "closed_ear_reference": closed,
            "asleep_ear_reference": closed,
            "partial_closure_normalized": partials[2],
            "reduced_reference_adjusted": bool(reduced_adjusted),
        }
        open_pitch = self.profiles["OPEN"].get("pitch")
        if open_pitch is not None:
            self.thresholds["open_pitch_reference"] = open_pitch["median"]
        self.static_ready = True
        self.last_failed_stage = None
        self.result_threshold = deep_threshold
        self.message = (
            "Etapas estaticas validas; referencia reducida ajustada "
            "automaticamente; inicia observacion natural de parpadeos"
            if reduced_adjusted else
            "Etapas estaticas validas; inicia observacion natural de parpadeos"
        )

    def _start_dynamic(self, mode):
        self.active = True
        self.dynamic_mode = str(mode).upper()
        self.active_profile = ("DYNAMIC_NATURAL" if self.dynamic_mode == "NATURAL"
                               else "DYNAMIC_VOLUNTARY")
        self.started = self.clock()
        self.capture_started = self.started + self.preparation_seconds
        self.preparing = self.preparation_seconds > 0.0
        self.progress = 0.0
        self.sample_attempts = 0
        self.dynamic_state = "OPEN"
        self.dynamic_event = None
        self.dynamic_invalid = False
        self.dynamic_closure = None
        self.dynamic_max_closure = 0.0
        self.message = self._stage_message(self.active_profile)

    def _update_dynamic(self, metrics):
        now = self.clock()
        if self._preparation_active(now):
            return None
        self.sample_attempts += 1
        duration = (float(self.config.get("natural_blink_observation_seconds", 60.0))
                    if self.dynamic_mode == "NATURAL" else
                    float(self.config.get("voluntary_blink_observation_seconds", 60.0)))
        self.progress = min(1.0, (now - self.started) / max(0.1, duration))
        self._update_dynamic_blink(metrics, now)
        events = (self.dynamic_natural_events if self.dynamic_mode == "NATURAL"
                  else self.dynamic_voluntary_events)
        required = (int(self.config.get("min_natural_blinks", 3))
                    if self.dynamic_mode == "NATURAL" else
                    int(self.config.get("min_voluntary_blinks", 5)))
        if self.dynamic_mode == "VOLUNTARY" and len(events) >= required:
            return self._advance_to_yawn("DYNAMIC_VOLUNTARY")
        if self.progress < 1.0:
            return None
        if self.dynamic_mode == "NATURAL":
            if len(events) >= required:
                return self._advance_to_yawn("DYNAMIC_NATURAL")
            if self.require_stage_confirmation:
                self.active = False
                self.active_profile = None
                self.message = ("Solo se detectaron %d parpadeos naturales validos. "
                                "%s" % (len(events),
                                          self._confirmation_message("DYNAMIC_VOLUNTARY")))
                return {"accepted": True, "profile": "DYNAMIC_NATURAL",
                        "dynamic_fallback": True, "natural_blinks": len(events),
                        "next_stage": "DYNAMIC_VOLUNTARY", "model_ready": False,
                        "reason": self.message}
            self._start_dynamic("VOLUNTARY")
            self.message = "Solo se detectaron %d parpadeos naturales validos. %s" % (
                len(events), self.INSTRUCTIONS["DYNAMIC_VOLUNTARY"])
            return {"accepted": True, "profile": "DYNAMIC_NATURAL",
                    "dynamic_fallback": True, "natural_blinks": len(events),
                    "model_ready": False, "reason": self.message}
        self.active = False
        self.active_profile = None
        self.last_failed_stage = "DYNAMIC_VOLUNTARY"
        self.message = ("Calibracion dinamica invalida: %d de %d parpadeos "
                        "voluntarios completos. Repita solo esta etapa." %
                        (len(events), required))
        result = {"accepted": False, "profile": "DYNAMIC_VOLUNTARY",
                  "failed_stage": "DYNAMIC_VOLUNTARY", "model_ready": False,
                  "reason": self.message}
        self.last_result = result
        return result

    def _advance_to_yawn(self, completed_stage):
        if self.require_stage_confirmation:
            self.active = False
            self.active_profile = None
            self.message = self._confirmation_message("YAWN")
            return {"accepted": True, "profile": completed_stage,
                    "next_stage": "YAWN", "model_ready": False,
                    "reason": self.message}
        self._start_yawn_calibration()
        return {"accepted": True, "profile": completed_stage,
                "yawn_started": True, "model_ready": False,
                "reason": self.message}

    def _start_yawn_calibration(self):
        self.yawn_only_recalibration = bool(
            self.model_ready and self.profile_data
            and not (self.dynamic_natural_events or self.dynamic_voluntary_events)
        )
        self.pending_recalibration = True
        self.active = True
        self.active_profile = "YAWN"
        self.dynamic_mode = None
        self.started = self.clock()
        self.capture_started = self.started + self.preparation_seconds
        self.preparing = self.preparation_seconds > 0.0
        self.progress = 0.0
        self.sample_attempts = 0
        self.sample_rejections = {}
        self.yawn_calibration_events = []
        self.yawn_calibration_state = "NORMAL"
        self.yawn_calibration_event = None
        self.yawn_calibration_mar = None
        self.yawn_calibration_max_mar = 0.0
        self.mouth_calibration = None
        self.message = self._stage_message("YAWN")

    def _update_yawn_calibration(self, metrics):
        now = self.clock()
        if self._preparation_active(now):
            return None
        self.sample_attempts += 1
        duration = float(self.config.get("yawn_calibration_seconds", 30.0))
        self.progress = min(1.0, (now - self.capture_started) / max(0.1, duration))
        self._update_yawn_calibration_cycle(metrics, now)
        required = int(self.config.get("min_calibration_yawns", 2))
        if len(self.yawn_calibration_events) >= required:
            error = self._build_mouth_calibration()
            if error is None:
                return (self._complete_yawn_recalibration()
                        if self.yawn_only_recalibration
                        else self._complete_profile())
            return self._fail_yawn_calibration(error)
        if self.progress < 1.0:
            return None
        return self._fail_yawn_calibration(
            "solo %d de %d bostezos completos" %
            (len(self.yawn_calibration_events), required)
        )

    def _update_yawn_calibration_cycle(self, metrics, now):
        if not metrics.get("face_detected", False):
            self._reject_sample("rostro no detectado durante bostezo")
            return
        if float(metrics.get("quality", 0.0)) < self.quality_threshold:
            self._reject_sample("calidad facial baja durante bostezo")
            return
        mar = metrics.get("mar")
        if mar is None or not np.isfinite(float(mar)):
            self._reject_sample("MAR no disponible")
            return
        mar = float(mar)
        self.yawn_calibration_mar = mar
        self.yawn_calibration_max_mar = max(self.yawn_calibration_max_mar, mar)
        baseline = self._mouth_baseline()
        start = baseline + float(self.config.get("yawn_calibration_start_gap", 0.10))
        returned = baseline + float(self.config.get("yawn_calibration_return_gap", 0.06))
        if self.yawn_calibration_state == "NORMAL":
            if mar >= start:
                self.yawn_calibration_state = "ABRIENDO"
                self.yawn_calibration_event = {
                    "started_at": now, "maximum_mar": mar, "samples": 1
                }
        elif self.yawn_calibration_state == "ABRIENDO":
            event = self.yawn_calibration_event
            event["maximum_mar"] = max(event["maximum_mar"], mar)
            event["samples"] += 1
            if mar <= returned:
                event["duration_seconds"] = max(0.0, now - event["started_at"])
                gap = event["maximum_mar"] - baseline
                minimum_gap = float(self.config.get("min_yawn_mar_gap", 0.15))
                minimum_duration = float(
                    self.config.get("min_calibration_yawn_seconds", 0.8)
                )
                if gap >= minimum_gap and event["duration_seconds"] >= minimum_duration:
                    self.yawn_calibration_events.append(event)
                else:
                    self._reject_sample("bostezo incompleto o apertura insuficiente")
                self.yawn_calibration_state = "NORMAL"
                self.yawn_calibration_event = None

    def _mouth_baseline(self):
        opened = self.profiles.get("OPEN", {}).get("mar", {})
        if opened.get("median") is not None:
            return float(opened["median"])
        mouth = (self.profile_data or {}).get("mouth", {})
        if mouth.get("baseline_mar") is not None:
            return float(mouth["baseline_mar"])
        return float(self.full_config.get("fatigue", {}).get(
            "mouth_closed_threshold", 0.38
        ))

    def _build_mouth_calibration(self):
        baseline = self._mouth_baseline()
        peaks = np.asarray([event["maximum_mar"]
                            for event in self.yawn_calibration_events], dtype=float)
        if peaks.size == 0:
            return "no hay bostezos validos"
        peak = float(np.median(peaks))
        gap = peak - baseline
        if gap < float(self.config.get("min_yawn_mar_gap", 0.15)):
            return "la apertura de la boca no se distingue del reposo"
        closed = baseline + 0.18 * gap
        opening = baseline + 0.38 * gap
        wide = baseline + 0.72 * gap
        self.thresholds.update({
            "mouth_closed_threshold": closed,
            "mouth_open_threshold": opening,
            "mouth_wide_threshold": wide,
        })
        self.mouth_calibration = {
            "baseline_mar": baseline, "peak_mar": peak,
            "peak_mad": float(np.median(np.abs(peaks - peak))),
            "event_count": int(peaks.size),
            "closed_threshold": closed, "open_threshold": opening,
            "wide_threshold": wide,
            "events": list(self.yawn_calibration_events),
        }
        return None

    def _complete_yawn_recalibration(self):
        profile = dict(self.profile_data)
        profile["thresholds"] = dict(profile.get("thresholds", {}))
        profile["thresholds"].update({
            key: self.thresholds[key] for key in (
                "mouth_closed_threshold", "mouth_open_threshold",
                "mouth_wide_threshold"
            )
        })
        profile["mouth"] = dict(self.mouth_calibration or {})
        profile["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        saved_profile = self._save_profile(profile)
        saved_parameters = self._save_parameters(self._parameter_export(profile))
        self.profile_data = profile
        self.model_ready = True
        self.pending_recalibration = False
        self.active = False
        self.active_profile = None
        self.last_failed_stage = None
        self.yawn_only_recalibration = False
        self.message = "Calibracion de bostezos completa; umbrales guardados"
        result = {"accepted": True, "profile": "YAWN",
                  "profile_data": profile, "thresholds": dict(self.thresholds),
                  "model_ready": True, "profile_saved": saved_profile,
                  "parameters_saved": saved_parameters, "reason": self.message}
        self.last_result = result
        return result

    def _fail_yawn_calibration(self, detail):
        self.active = False
        self.active_profile = None
        self.last_failed_stage = "YAWN"
        self.message = "Calibracion de bostezos invalida: %s. Repita esta etapa." % detail
        result = {"accepted": False, "profile": "YAWN",
                  "failed_stage": "YAWN", "model_ready": False,
                  "reason": self.message}
        self.last_result = result
        return result

    def _stage_message(self, profile):
        instruction = self.INSTRUCTIONS.get(profile, "Siga la instruccion de calibracion")
        if self.preparing:
            return "Preparese para %s: %s" % (
                self.PROFILE_LABELS.get(profile, profile), instruction,
            )
        return instruction

    def _confirmation_message(self, profile):
        label = self.PROFILE_LABELS.get(profile, profile)
        instruction = self.INSTRUCTIONS.get(profile, "Siga la instruccion de calibracion")
        return "Haga click o presione ESPACIO para iniciar %s. %s" % (
            label, instruction,
        )

    def _preparation_active(self, now):
        if not self.preparing:
            return False
        if now < self.capture_started:
            remaining = max(0.0, self.capture_started - now)
            self.progress = 0.0
            self.message = "Preparese para %s en %.1f s: %s" % (
                self.PROFILE_LABELS.get(self.active_profile, self.active_profile),
                remaining,
                self.INSTRUCTIONS.get(self.active_profile, "Siga la instruccion"),
            )
            return True
        self.preparing = False
        self.started = now
        self.capture_started = now
        self.samples = []
        self.sample_attempts = 0
        self.sample_rejections = {}
        self.progress = 0.0
        self.dynamic_state = "OPEN"
        self.dynamic_event = None
        self.dynamic_invalid = False
        self.message = self.INSTRUCTIONS.get(
            self.active_profile, "Siga la instruccion de calibracion"
        )
        return False

    def _update_dynamic_blink(self, metrics, now):
        sample = self._extract_sample(
            metrics, require_quality=True, allow_closed_eyes=True
        )
        # Un cuadro perdido o un salto aislado de pose no invalida un ciclo
        # ocular completo; simplemente no aporta una transicion.
        if sample is None:
            return
        closure = self._normalized_from_sample(sample)
        if closure is None:
            return
        self.dynamic_closure = float(closure)
        self.dynamic_max_closure = max(self.dynamic_max_closure, float(closure))
        start_level = float(self.config.get("blink_start_closure_level", 0.15))
        closed_level = float(self.config.get("blink_closed_closure_level", 0.35))
        reopen_level = float(self.config.get("blink_reopen_level", 0.15))
        if self.dynamic_state == "OPEN":
            if closure >= start_level:
                self.dynamic_state = "CLOSING"
                self.dynamic_event = {"started_at": now, "closed_at": None,
                    "opening_at": None, "minimum_ear": sample["ear"],
                    "maximum_closure": closure}
                self.dynamic_invalid = False
        elif self.dynamic_state == "CLOSING":
            self._dynamic_extreme(sample, closure)
            if closure >= closed_level:
                self.dynamic_state = "CLOSED"
                self.dynamic_event["closed_at"] = now
            elif closure <= reopen_level:
                self._reset_dynamic_cycle(rejected=True)
        elif self.dynamic_state == "CLOSED":
            self._dynamic_extreme(sample, closure)
            if closure < closed_level:
                self.dynamic_state = "OPENING"
                self.dynamic_event["opening_at"] = now
        elif self.dynamic_state == "OPENING":
            self._dynamic_extreme(sample, closure)
            if closure >= closed_level:
                self.dynamic_state = "CLOSED"
                self.dynamic_event["opening_at"] = None
            elif closure <= reopen_level:
                self._complete_dynamic_blink(now)

    def _dynamic_head_moved(self, sample):
        open_summary = self.profiles.get("OPEN", {})
        maximum = float(self.config.get("max_dynamic_head_delta_degrees", 12.0))
        for key in ("pitch", "yaw", "roll"):
            if key in sample and key in open_summary:
                if abs(float(sample[key]) - float(open_summary[key]["median"])) > maximum:
                    return True
        return False

    def _normalized_from_sample(self, sample):
        opened = self.profiles["OPEN"]["ear"]["median"]
        closed = self.profiles["CLOSED"]["ear"]["median"]
        denominator = opened - closed
        if denominator <= 1e-6:
            return None
        return max(0.0, min(1.0, (opened - float(sample["ear"])) / denominator))

    def _dynamic_extreme(self, sample, closure):
        if self.dynamic_event is not None:
            self.dynamic_event["minimum_ear"] = min(
                self.dynamic_event["minimum_ear"], sample["ear"])
            self.dynamic_event["maximum_closure"] = max(
                self.dynamic_event["maximum_closure"], closure)

    def _complete_dynamic_blink(self, now):
        event = self.dynamic_event
        if event is None or event.get("closed_at") is None or self.dynamic_invalid:
            self._reset_dynamic_cycle(rejected=True)
            return
        duration = max(0.0, now - event["started_at"])
        minimum = float(self.config.get("blink_min_seconds", 0.08))
        maximum = float(self.config.get("calibration_blink_max_seconds", 1.2))
        if minimum <= duration <= maximum:
            event["duration_seconds"] = duration
            event["descent_seconds"] = max(0.0, event["closed_at"] - event["started_at"])
            event["reopen_seconds"] = max(0.0, now - (
                event["opening_at"] if event.get("opening_at") is not None
                else event["closed_at"]))
            target = (self.dynamic_natural_events if self.dynamic_mode == "NATURAL"
                      else self.dynamic_voluntary_events)
            target.append(event)
            self._reset_dynamic_cycle(rejected=False)
        else:
            self._reset_dynamic_cycle(rejected=True)

    def _reset_dynamic_cycle(self, rejected):
        if rejected:
            self.dynamic_rejected_events += 1
        self.dynamic_state = "OPEN"
        self.dynamic_event = None
        self.dynamic_invalid = False

    def _complete_profile(self):
        natural_stats = self._blink_statistics(self.dynamic_natural_events, "natural")
        voluntary_stats = self._blink_statistics(self.dynamic_voluntary_events, "voluntary")
        min_natural = int(self.config.get("min_natural_blinks", 3))
        selected = natural_stats if natural_stats["event_count"] >= min_natural else voluntary_stats
        if selected["event_count"] <= 0:
            return None
        eyes = {}
        for side, feature in (("left", "left_ear"), ("right", "right_ear"),
                              ("combined", "ear")):
            eyes[side] = {
                "open": self.profiles["OPEN"][feature]["median"],
                "reduced": self.profiles["REDUCED"][feature]["median"],
                "closed": self.profiles["CLOSED"][feature]["median"],
                "open_variation": self.profiles["OPEN"][feature]["robust_std"],
                "reduced_variation": self.profiles["REDUCED"][feature]["robust_std"],
                "closed_variation": self.profiles["CLOSED"][feature]["robust_std"],
            }
        neutral = {}
        for feature in ("pitch", "yaw", "roll"):
            stats = self.profiles["OPEN"].get(feature)
            neutral[feature] = stats["median"] if stats else 0.0
        quality = dict((name, dict(self.profiles[name]["quality"]))
                       for name in self.STATIC_PROFILES)
        quality["dynamic"] = {
            "natural_events": len(self.dynamic_natural_events),
            "voluntary_events": len(self.dynamic_voluntary_events),
            "rejected_events": self.dynamic_rejected_events,
            "selected_source": selected["source"],
        }
        quality["yawn"] = {
            "events": len(self.yawn_calibration_events),
            "rejections": dict(self.sample_rejections),
        }
        profile = {
            "format_version": self.FORMAT_VERSION,
            "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "calibrated_monotonic_seconds": self.clock(),
            "eyes": eyes,
            "partial_closure_normalized": self.thresholds["partial_closure_normalized"],
            "blink_baseline": {"natural": natural_stats, "voluntary": voluntary_stats,
                               "selected": selected},
            "mouth": dict(self.mouth_calibration or {}),
            "neutral_head_pose": neutral, "quality": quality,
            "thresholds": dict(self.thresholds), "stages": dict(self.profiles),
        }
        parameter_export = self._parameter_export(profile)
        saved_profile = self._save_profile(profile)
        saved_parameters = self._save_parameters(parameter_export)
        self.profile_data = profile
        self.model_ready = True
        self.pending_recalibration = False
        self.active = False
        self.active_profile = None
        self.dynamic_mode = None
        self.last_failed_stage = None
        saved = bool(saved_profile and saved_parameters)
        if saved:
            self.message = "Calibracion completa; perfil y parametros guardados"
        elif saved_profile:
            self.message = "Calibracion completa; perfil guardado, parametros no guardados"
        elif saved_parameters:
            self.message = "Calibracion completa; parametros guardados, perfil no guardado"
        else:
            self.message = "Calibracion completa; no se pudo guardar el perfil"
        result = {"accepted": True, "profile": "COMPLETE", "profile_data": profile,
                  "profiles": dict(self.profiles), "thresholds": dict(self.thresholds),
                  "static_ready": True, "model_ready": True,
                  "profile_saved": saved_profile,
                  "parameters_saved": saved_parameters,
                  "parameters_path": self.parameters_path,
                  "reason": self.message}
        self.last_result = result
        return result

    def _parameter_export(self, profile):
        keys = (
            "duration_seconds", "preparation_seconds",
            "require_stage_confirmation", "min_samples",
            "min_valid_sample_ratio", "min_ear_gap", "min_open_closed_gap",
            "natural_blink_observation_seconds",
            "voluntary_blink_observation_seconds", "min_natural_blinks",
            "min_voluntary_blinks", "calibration_blink_max_seconds",
            "yawn_calibration_seconds", "min_calibration_yawns",
            "min_calibration_yawn_seconds", "min_yawn_mar_gap",
        )
        calibration_settings = {}
        for key in keys:
            if key in self.config:
                calibration_settings[key] = self.config[key]
        return {
            "format_version": self.FORMAT_VERSION,
            "kind": "tt2_calibration_parameters",
            "profile_path": self.profile_path,
            "calibrated_at": profile.get("calibrated_at"),
            "eyes": profile.get("eyes", {}),
            "partial_closure_normalized": profile.get(
                "partial_closure_normalized"
            ),
            "blink_baseline": profile.get("blink_baseline", {}),
            "mouth": profile.get("mouth", {}),
            "neutral_head_pose": profile.get("neutral_head_pose", {}),
            "quality": profile.get("quality", {}),
            "thresholds": profile.get("thresholds", {}),
            "calibration_settings": calibration_settings,
            "fatigue_settings": dict(self.full_config.get("fatigue", {})),
            "profile_data": profile,
        }

    @staticmethod
    def _blink_statistics(events, source):
        durations = np.asarray([e["duration_seconds"] for e in events], dtype=float)
        if durations.size == 0:
            return {"source": source, "event_count": 0, "median_seconds": None,
                    "p75_seconds": None, "p90_seconds": None, "mad_seconds": None,
                    "std_seconds": None}
        median = float(np.median(durations))
        return {
            "source": source, "event_count": int(durations.size),
            "median_seconds": median, "p75_seconds": float(np.percentile(durations, 75)),
            "p90_seconds": float(np.percentile(durations, 90)),
            "mad_seconds": float(np.median(np.abs(durations - median))),
            "std_seconds": float(np.std(durations)),
            "median_descent_seconds": float(np.median([e["descent_seconds"] for e in events])),
            "median_reopen_seconds": float(np.median([e["reopen_seconds"] for e in events])),
            "median_minimum_ear": float(np.median([e["minimum_ear"] for e in events])),
        }

    def _save_profile(self, profile):
        return self._save_json_atomic(self.profile_path, profile)

    def _save_parameters(self, parameters):
        return self._save_json_atomic(self.parameters_path, parameters)

    def _save_json_atomic(self, path, payload):
        path = os.path.abspath(path)
        directory, temporary = os.path.dirname(path), path + ".tmp"
        try:
            if directory and not os.path.isdir(directory):
                os.makedirs(directory)
            with open(temporary, "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return True
        except Exception as exc:
            self.message = "No se pudo guardar el perfil: %s" % exc
            try:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            except Exception:
                pass
            return False

    def _load_profile(self):
        errors = []
        for path, from_parameters in (
            (self.profile_path, False),
            (self.parameters_path, True),
        ):
            absolute = os.path.abspath(path)
            if not os.path.exists(absolute):
                continue
            try:
                with open(absolute, "r") as handle:
                    loaded = json.load(handle)
                if from_parameters:
                    loaded = loaded.get("profile_data", loaded)
                migrated = self._migrate_profile(loaded)
                if not self._profile_valid(migrated):
                    errors.append("perfil invalido en %s" % path)
                    continue
                self._apply_loaded_profile(migrated, from_parameters)
                return
            except Exception as exc:
                errors.append("%s: %s" % (path, exc))
        if errors:
            self.message = "No se pudo cargar perfil persistente: %s" % "; ".join(errors)

    def _apply_loaded_profile(self, profile, from_parameters):
        self.profile_data = profile
        self.profiles = dict(profile.get("stages", {}))
        self.thresholds = dict(profile.get("thresholds", {}))
        self.static_ready = True
        self.model_ready = True
        self.pending_recalibration = False
        self.message = ("Parametros de calibracion cargados" if from_parameters
                        else "Perfil de calibracion cargado")

    def _migrate_profile(self, profile):
        if int(profile.get("format_version", 1)) >= self.FORMAT_VERSION:
            return profile
        legacy_profiles = profile.get("profiles", profile.get("stages", {}))
        mapped = {}
        for old_name, new_name in (("OPEN", "OPEN"), ("DROWSY", "REDUCED"),
                                   ("ASLEEP", "CLOSED")):
            if old_name in legacy_profiles:
                stage = dict(legacy_profiles[old_name])
                if "ear" in stage:
                    stage.setdefault("left_ear", dict(stage["ear"]))
                    stage.setdefault("right_ear", dict(stage["ear"]))
                mapped[new_name] = stage
        if len(mapped) != 3:
            return profile
        eyes = {}
        for side, feature in (("left", "left_ear"), ("right", "right_ear"),
                              ("combined", "ear")):
            if any(feature not in mapped[name] for name in self.STATIC_PROFILES):
                feature = "ear"
            eyes[side] = {"open": mapped["OPEN"][feature]["median"],
                          "reduced": mapped["REDUCED"][feature]["median"],
                          "closed": mapped["CLOSED"][feature]["median"]}
        opened, reduced = eyes["combined"]["open"], eyes["combined"]["reduced"]
        closed = eyes["combined"]["closed"]
        blink = profile.get("blink_baseline")
        if not blink:
            return profile
        return {
            "format_version": self.FORMAT_VERSION,
            "calibrated_at": profile.get("calibrated_at", "legacy"),
            "eyes": eyes,
            "partial_closure_normalized": ((opened - reduced) / max(1e-6, opened - closed)),
            "blink_baseline": blink,
            "neutral_head_pose": profile.get("neutral_head_pose",
                {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}),
            "quality": profile.get("quality", {"migration": "legacy_v1"}),
            "thresholds": profile.get("thresholds", {}),
            "stages": mapped, "migrated_from_version": 1,
        }

    def _profile_valid(self, profile):
        if not isinstance(profile, dict):
            return False
        if int(profile.get("format_version", 0)) != self.FORMAT_VERSION:
            return False
        eyes = profile.get("eyes", {})
        minimum_ear = float(self.config.get("min_possible_ear", 0.03))
        maximum_ear = float(self.config.get("max_possible_ear", 0.60))
        for side in ("left", "right", "combined"):
            ref = eyes.get(side, {})
            opened, reduced, closed = ref.get("open"), ref.get("reduced"), ref.get("closed")
            if None in (opened, reduced, closed):
                return False
            values = [float(opened), float(reduced), float(closed)]
            if not all(np.isfinite(value) and minimum_ear <= value <= maximum_ear
                       for value in values):
                return False
            if not (values[0] - values[1] >= self.min_ear_gap
                    and values[1] - values[2] >= self.min_ear_gap
                    and values[0] - values[2] >= self.min_open_closed_gap):
                return False
        partial = profile.get("partial_closure_normalized")
        if partial is None or not np.isfinite(float(partial)):
            return False
        if not 0.0 < float(partial) < 1.0:
            return False
        blink = profile.get("blink_baseline", {})
        selected = blink.get("selected", blink)
        median = selected.get("median_seconds")
        source = str(selected.get("source", "natural")).lower()
        required = int(self.config.get(
            "min_voluntary_blinks" if source == "voluntary"
            else "min_natural_blinks",
            5 if source == "voluntary" else 3,
        ))
        return bool(
            int(selected.get("event_count", 0)) >= required
            and median is not None
            and np.isfinite(float(median))
            and float(self.config.get("blink_min_seconds", 0.08))
            <= float(median)
            <= float(self.config.get("calibration_blink_max_seconds", 1.2))
        )
