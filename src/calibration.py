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
    }
    PROFILE_KEYS = {"OPEN": "O", "REDUCED": "R", "CLOSED": "C"}
    INSTRUCTIONS = {
        "OPEN": ("Mire al frente con postura natural; no abra exageradamente "
                 "los ojos y evite mover la cabeza"),
        "REDUCED": ("Mantenga los parpados parcialmente cerrados, sin cerrarlos "
                    "por completo"),
        "CLOSED": ("Cierre los ojos de forma natural; haga dos cierres si la "
                   "deteccion facial lo permite"),
        "DYNAMIC_NATURAL": "Mire al frente de manera natural durante unos segundos",
        "DYNAMIC_VOLUNTARY": ("Realice entre cinco y ocho parpadeos normales, "
                              "sin exagerarlos"),
    }
    FEATURES = ("ear", "left_ear", "right_ear", "mar", "pitch", "yaw", "roll")

    def __init__(self, config, clock=None):
        self.full_config = config
        self.config = config.get("calibration", {})
        self.clock = clock or time.monotonic
        self.duration = float(self.config.get("duration_seconds", 5.0))
        self.min_samples = int(self.config.get("min_samples", 30))
        self.quality_threshold = float(self.config.get("quality_threshold", 0.30))
        self.min_valid_ratio = float(self.config.get("min_valid_sample_ratio", 0.65))
        self.min_ear_gap = float(self.config.get("min_ear_gap", 0.020))
        self.min_open_closed_gap = float(self.config.get("min_open_closed_gap", 0.060))
        self.max_ear_std = float(self.config.get("max_ear_std", 0.035))
        self.max_ear_mad = float(self.config.get("max_ear_mad", 0.025))
        self.max_head_std = float(self.config.get("max_head_angle_std_degrees", 6.0))
        self.max_eye_asymmetry_ratio = float(
            self.config.get("max_eye_asymmetry_ratio", 0.35)
        )
        self.max_partial_eye_difference = float(
            self.config.get("max_partial_eye_difference", 0.25)
        )
        self.profile_path = self.config.get("profile_path", "calibration_profile.json")

        self.active = False
        self.active_profile = None
        self.started = 0.0
        self.samples = []
        self.sample_attempts = 0
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
        self.dynamic_rejected_events = 0
        self._load_profile()

    def ready_for_monitoring(self):
        return bool(self.model_ready and self.profile_data and not self.active
                    and not self.pending_recalibration)

    def next_required_stage(self):
        for profile in self.STATIC_PROFILES:
            if profile not in self.profiles:
                return profile
        return None

    def start(self, profile="OPEN"):
        requested = str(profile or "OPEN").upper()
        if requested in ("FULL", "SEQUENCE", "CALIBRATION"):
            self.profiles = {}
            self.thresholds = {}
            self.static_ready = False
            self.dynamic_natural_events = []
            self.dynamic_voluntary_events = []
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

        profile = self.PROFILE_ALIASES.get(requested)
        if profile not in self.STATIC_PROFILES:
            raise ValueError("Perfil de calibracion desconocido: %s" % requested)
        self.pending_recalibration = True
        self.active = True
        self.active_profile = profile
        self.started = self.clock()
        self.samples = []
        self.sample_attempts = 0
        self.progress = 0.0
        self.last_result = None
        self.last_failed_stage = None
        self.message = self.INSTRUCTIONS[profile]

    def update(self, metrics):
        if not self.active:
            return None
        if self.active_profile in ("DYNAMIC_NATURAL", "DYNAMIC_VOLUNTARY"):
            return self._update_dynamic(metrics)

        now = self.clock()
        self.sample_attempts += 1
        self.progress = min(1.0, (now - self.started) / max(0.1, self.duration))
        sample = self._extract_sample(metrics, require_quality=True)
        if sample is not None:
            self.samples.append(sample)
        if self.progress < 1.0:
            return None

        profile = self.active_profile
        self.active = False
        self.active_profile = None
        summary, error = self._summarize(self.samples, self.sample_attempts)
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
            self._start_dynamic("NATURAL")
            result["dynamic_started"] = True
            result["reason"] = self.message
        self.last_result = result
        return result

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
        return {
            "calibration_active": self.active,
            "calibration_target": self.active_profile or "",
            "calibration_progress": self.progress if self.active else 0.0,
            "calibration_ready": self.ready_for_monitoring(),
            "calibration_static_ready": self.static_ready,
            "calibration_profiles": " ".join(status),
            "calibration_message": self.message,
            "calibration_failed_stage": self.last_failed_stage or "",
            "calibration_valid_samples": (len(self.samples)
                if self.active_profile in self.STATIC_PROFILES else 0),
            "calibration_sample_attempts": self.sample_attempts,
            "calibration_blink_count": blink_count,
            "calibration_natural_blinks": len(self.dynamic_natural_events),
            "calibration_voluntary_blinks": len(self.dynamic_voluntary_events),
            "calibrated_ear_threshold": self.thresholds.get("ear_threshold"),
            "calibrated_ear_open_threshold": self.thresholds.get("ear_open_threshold"),
            "reduced_ear_threshold": self.thresholds.get("reduced_ear_threshold"),
            "calibration_format_version": (self.profile_data.get("format_version")
                if self.profile_data else None),
        }

    def _extract_sample(self, metrics, require_quality):
        if not metrics.get("face_detected", False):
            return None
        quality = float(metrics.get("quality", 0.0))
        if require_quality and quality < self.quality_threshold:
            return None
        if require_quality and metrics.get("pose_reliable") is False:
            return None
        fallback_eye = metrics.get("ear")
        left_ear = metrics.get("left_ear", fallback_eye)
        right_ear = metrics.get("right_ear", fallback_eye)
        fallback_reliable = bool(metrics.get("eye_reliable", True))
        if not bool(metrics.get("left_eye_reliable", fallback_reliable)):
            return None
        if not bool(metrics.get("right_eye_reliable", fallback_reliable)):
            return None
        if not self._valid_ear(left_ear) or not self._valid_ear(right_ear):
            return None
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

    def _summarize(self, samples, attempts):
        if len(samples) < self.min_samples:
            return None, "solo %d muestras validas; se requieren %d" % (
                len(samples), self.min_samples)
        valid_ratio = len(samples) / float(max(1, attempts))
        if valid_ratio < self.min_valid_ratio:
            return None, "cobertura valida %.0f%% inferior al minimo %.0f%%" % (
                valid_ratio * 100.0, self.min_valid_ratio * 100.0)
        summary = {"sample_count": len(samples), "attempted_samples": int(attempts),
                   "valid_sample_ratio": valid_ratio}
        for feature in self.FEATURES:
            values = [sample[feature] for sample in samples if feature in sample]
            if values:
                summary[feature] = self._statistics(values)
        if "ear" not in summary:
            return None, "EAR no disponible"
        for feature in ("ear", "left_ear", "right_ear"):
            stats = summary.get(feature)
            if stats is None:
                return None, "faltan mediciones de ambos ojos"
            if stats["robust_std"] > self.max_ear_std or stats["mad"] > self.max_ear_mad:
                return None, "variacion ocular excesiva en %s" % feature
        for feature in ("pitch", "yaw", "roll"):
            stats = summary.get(feature)
            if stats is not None and stats["std"] > self.max_head_std:
                return None, "movimiento excesivo de cabeza en %s" % feature
        left, right = summary["left_ear"]["median"], summary["right_ear"]["median"]
        asymmetry = abs(left - right) / max(1e-6, (left + right) * 0.5)
        if asymmetry > self.max_eye_asymmetry_ratio:
            return None, "diferencia entre ojos %.0f%% superior al maximo" % (
                asymmetry * 100.0)
        summary["eye_asymmetry_ratio"] = asymmetry
        summary["quality"] = {
            "median": float(np.median([sample["quality"] for sample in samples])),
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
        invalid = set()
        for feature in ("ear", "left_ear", "right_ear"):
            opened = self.profiles["OPEN"][feature]["median"]
            reduced = self.profiles["REDUCED"][feature]["median"]
            closed = self.profiles["CLOSED"][feature]["median"]
            if opened - reduced < self.min_ear_gap:
                invalid.update(("OPEN", "REDUCED"))
            if reduced - closed < self.min_ear_gap:
                invalid.update(("REDUCED", "CLOSED"))
            if opened - closed < self.min_open_closed_gap:
                invalid.update(("OPEN", "CLOSED"))
        if invalid:
            self.static_ready = False
            self.thresholds = {}
            self.last_failed_stage = "REDUCED" if "REDUCED" in invalid else sorted(invalid)[0]
            # La etapa señalada deja de considerarse válida y nunca llega al
            # perfil persistente; las demás muestras se conservan para repetir
            # sólo la captura afectada.
            self.profiles.pop(self.last_failed_stage, None)
            self.message = ("Geometria no separable: debe cumplirse ABIERTO > REDUCIDO > "
                            "CERRADO con los margenes configurados. Repita: %s" % ", ".join(
                                self.PROFILE_LABELS[name] for name in self.STATIC_PROFILES
                                if name in invalid))
            return
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
        }
        open_pitch = self.profiles["OPEN"].get("pitch")
        if open_pitch is not None:
            self.thresholds["open_pitch_reference"] = open_pitch["median"]
        self.static_ready = True
        self.last_failed_stage = None
        self.result_threshold = deep_threshold
        self.message = "Etapas estaticas validas; inicia observacion natural de parpadeos"

    def _start_dynamic(self, mode):
        self.active = True
        self.dynamic_mode = str(mode).upper()
        self.active_profile = ("DYNAMIC_NATURAL" if self.dynamic_mode == "NATURAL"
                               else "DYNAMIC_VOLUNTARY")
        self.started = self.clock()
        self.progress = 0.0
        self.sample_attempts = 0
        self.dynamic_state = "OPEN"
        self.dynamic_event = None
        self.dynamic_invalid = False
        self.message = self.INSTRUCTIONS[self.active_profile]

    def _update_dynamic(self, metrics):
        now = self.clock()
        self.sample_attempts += 1
        duration = (float(self.config.get("natural_blink_observation_seconds", 8.0))
                    if self.dynamic_mode == "NATURAL" else
                    float(self.config.get("voluntary_blink_observation_seconds", 15.0)))
        self.progress = min(1.0, (now - self.started) / max(0.1, duration))
        self._update_dynamic_blink(metrics, now)
        events = (self.dynamic_natural_events if self.dynamic_mode == "NATURAL"
                  else self.dynamic_voluntary_events)
        required = (int(self.config.get("min_natural_blinks", 3))
                    if self.dynamic_mode == "NATURAL" else
                    int(self.config.get("min_voluntary_blinks", 5)))
        if self.dynamic_mode == "VOLUNTARY" and len(events) >= required:
            return self._complete_profile()
        if self.progress < 1.0:
            return None
        if self.dynamic_mode == "NATURAL":
            if len(events) >= required:
                return self._complete_profile()
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

    def _update_dynamic_blink(self, metrics, now):
        sample = self._extract_sample(metrics, require_quality=True)
        if sample is None or self._dynamic_head_moved(sample):
            if self.dynamic_state != "OPEN":
                self.dynamic_invalid = True
            return
        closure = self._normalized_from_sample(sample)
        if closure is None:
            return
        start_level = float(self.config.get("blink_start_closure_level", 0.35))
        closed_level = float(self.config.get("blink_closed_closure_level", 0.75))
        reopen_level = float(self.config.get("blink_reopen_level", 0.25))
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
        profile = {
            "format_version": self.FORMAT_VERSION,
            "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "calibrated_monotonic_seconds": self.clock(),
            "eyes": eyes,
            "partial_closure_normalized": self.thresholds["partial_closure_normalized"],
            "blink_baseline": {"natural": natural_stats, "voluntary": voluntary_stats,
                               "selected": selected},
            "neutral_head_pose": neutral, "quality": quality,
            "thresholds": dict(self.thresholds), "stages": dict(self.profiles),
        }
        saved = self._save_profile(profile)
        self.profile_data = profile
        self.model_ready = True
        self.pending_recalibration = False
        self.active = False
        self.active_profile = None
        self.dynamic_mode = None
        self.last_failed_stage = None
        self.message = ("Calibracion completa y perfil guardado" if saved
                        else "Calibracion completa; no se pudo guardar el perfil")
        result = {"accepted": True, "profile": "COMPLETE", "profile_data": profile,
                  "profiles": dict(self.profiles), "thresholds": dict(self.thresholds),
                  "static_ready": True, "model_ready": True,
                  "profile_saved": saved, "reason": self.message}
        self.last_result = result
        return result

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
        path = os.path.abspath(self.profile_path)
        directory, temporary = os.path.dirname(path), path + ".tmp"
        try:
            if directory and not os.path.isdir(directory):
                os.makedirs(directory)
            with open(temporary, "w") as handle:
                json.dump(profile, handle, indent=2, sort_keys=True)
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
        path = os.path.abspath(self.profile_path)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as handle:
                loaded = json.load(handle)
            migrated = self._migrate_profile(loaded)
            if not self._profile_valid(migrated):
                self.message = "Perfil existente invalido; se requiere recalibracion"
                return
            self.profile_data = migrated
            self.profiles = dict(migrated.get("stages", {}))
            self.thresholds = dict(migrated.get("thresholds", {}))
            self.static_ready = True
            self.model_ready = True
            self.pending_recalibration = False
            self.message = "Perfil de calibracion cargado"
        except Exception as exc:
            self.message = "No se pudo cargar el perfil: %s" % exc

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
