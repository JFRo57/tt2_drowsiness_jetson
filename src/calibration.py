import time

import numpy as np


class CalibrationManager(object):
    PROFILES = ("OPEN", "DROWSY", "ASLEEP")
    PROFILE_LABELS = {
        "OPEN": "OJOS ABIERTOS",
        "DROWSY": "POSIBLE SOMNOLENCIA",
        "ASLEEP": "DORMIDO",
    }
    PROFILE_KEYS = {"OPEN": "O", "DROWSY": "S", "ASLEEP": "D"}
    INSTRUCTIONS = {
        "OPEN": "Mire al frente y mantenga los ojos bien abiertos",
        "DROWSY": "Simule somnolencia: ojos entrecerrados y postura relajada",
        "ASLEEP": "Cierre los ojos y adopte su postura habitual de dormido",
    }
    FEATURES = ("ear", "mar", "pitch", "yaw", "roll")
    DEFAULT_SCALES = {
        "ear": 0.04,
        "mar": 0.15,
        "pitch": 12.0,
        "yaw": 15.0,
        "roll": 15.0,
    }
    FEATURE_WEIGHTS = {
        "ear": 2.5,
        "mar": 0.35,
        "pitch": 0.8,
        "yaw": 0.55,
        "roll": 0.55,
    }

    def __init__(self, config, clock=None):
        self.full_config = config
        self.config = config.get("calibration", {})
        self.clock = clock or time.monotonic
        self.duration = float(self.config.get("duration_seconds", 4.0))
        self.min_samples = int(self.config.get("min_samples", 12))
        self.quality_threshold = float(self.config.get("quality_threshold", 0.30))
        self.min_ear_gap = float(self.config.get("min_ear_gap", 0.015))
        self.max_ear_std = float(self.config.get("max_ear_std", 0.08))
        self.max_profile_distance = float(self.config.get("max_profile_distance", 4.0))
        self.feature_scales = dict(self.DEFAULT_SCALES)
        self.feature_scales.update(self.config.get("feature_scales", {}))

        self.active = False
        self.active_profile = None
        self.started = 0.0
        self.samples = []
        self.progress = 0.0
        self.profiles = {}
        self.thresholds = {}
        self.model_ready = False
        self.result_threshold = None
        self.last_result = None
        self.message = "Calibracion inactiva"

    def start(self, profile="OPEN"):
        profile = str(profile or "OPEN").upper()
        if profile not in self.PROFILES:
            raise ValueError("Perfil de calibracion desconocido: %s" % profile)
        self.active = True
        self.active_profile = profile
        self.started = self.clock()
        self.samples = []
        self.progress = 0.0
        self.result_threshold = None
        self.last_result = None
        self.message = self.INSTRUCTIONS[profile]

    def update(self, metrics):
        if not self.active:
            return None
        now = self.clock()
        self.progress = min(1.0, (now - self.started) / max(0.1, self.duration))
        sample = self._extract_sample(metrics, require_quality=True)
        if sample is not None:
            self.samples.append(sample)
        if self.progress < 1.0:
            return None

        profile = self.active_profile
        self.active = False
        self.active_profile = None
        summary = self._summarize(self.samples)
        if summary is None:
            self.message = "Calibracion %s invalida: muestras insuficientes o inestables" % self.PROFILE_LABELS[profile]
            return None

        self.profiles[profile] = summary
        self._rebuild_model()
        result = {
            "profile": profile,
            "summary": summary,
            "profiles": dict(self.profiles),
            "thresholds": dict(self.thresholds),
            "model_ready": self.model_ready,
        }

        if profile == "OPEN" and not self.model_ready:
            legacy_threshold = max(0.15, min(0.32, summary["ear"]["median"] * 0.72))
            result["ear_threshold"] = legacy_threshold
            self.result_threshold = legacy_threshold
        elif self.model_ready:
            self.result_threshold = self.thresholds["ear_threshold"]

        self.last_result = result
        return result

    def classify(self, metrics):
        if not self.model_ready:
            return {"profile": "NO_CALIBRADO", "confidence": 0.0, "scores": {}}
        sample = self._extract_sample(metrics, require_quality=True)
        if sample is None:
            return {"profile": "DESCONOCIDO", "confidence": 0.0, "scores": {}}

        scores = {}
        for profile in self.PROFILES:
            summary = self.profiles[profile]
            weighted_total = 0.0
            weight_sum = 0.0
            for feature in self.FEATURES:
                if feature not in sample or feature not in summary:
                    continue
                weight = float(self.FEATURE_WEIGHTS[feature])
                fixed_scale = max(1e-6, float(self.feature_scales[feature]))
                observed_scale = float(summary[feature].get("std", 0.0)) * 2.0
                scale = max(fixed_scale, observed_scale)
                delta = (float(sample[feature]) - float(summary[feature]["median"])) / scale
                weighted_total += weight * delta * delta
                weight_sum += weight
            if sample.get("gaze") and summary.get("gaze") and sample["gaze"] != summary["gaze"]:
                weighted_total += 0.15
                weight_sum += 0.15
            scores[profile] = (weighted_total / weight_sum) ** 0.5 if weight_sum else float("inf")

        ordered = sorted(scores.items(), key=lambda item: item[1])
        best_profile, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else self.max_profile_distance
        if best_score > self.max_profile_distance:
            return {"profile": "DESCONOCIDO", "confidence": 0.0, "scores": scores}
        absolute_confidence = max(0.0, 1.0 - (best_score / self.max_profile_distance))
        margin_confidence = max(0.0, (second_score - best_score) / max(second_score, 1e-6))
        confidence = min(1.0, (absolute_confidence + margin_confidence) / 2.0)
        return {"profile": best_profile, "confidence": confidence, "scores": scores}

    def status_metrics(self):
        status = []
        for profile in self.PROFILES:
            status.append("%s:%s" % (self.PROFILE_KEYS[profile], "OK" if profile in self.profiles else "--"))
        return {
            "calibration_active": self.active,
            "calibration_target": self.active_profile or "",
            "calibration_progress": self.progress if self.active else 0.0,
            "calibration_ready": self.model_ready,
            "calibration_profiles": " ".join(status),
            "drowsy_ear_threshold": self.thresholds.get("drowsy_ear_threshold"),
            "calibrated_ear_threshold": self.thresholds.get("ear_threshold"),
        }

    def _extract_sample(self, metrics, require_quality):
        if not metrics.get("face_detected", False):
            return None
        quality = float(metrics.get("quality", 0.0))
        if require_quality and quality < self.quality_threshold:
            return None
        ear = metrics.get("ear")
        if ear is None or float(ear) <= 0.0:
            return None
        sample = {"ear": float(ear), "quality": quality, "gaze": metrics.get("gaze", "DESCONOCIDA")}
        for feature in ("mar", "pitch", "yaw", "roll"):
            value = metrics.get(feature)
            if value is not None:
                sample[feature] = float(value)
        return sample

    def _summarize(self, samples):
        if len(samples) < self.min_samples:
            return None
        summary = {"sample_count": len(samples)}
        for feature in self.FEATURES:
            values = [sample[feature] for sample in samples if feature in sample]
            if not values:
                continue
            arr = np.array(values, dtype=float)
            summary[feature] = {
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
            }
        if "ear" not in summary:
            return None
        ear_median = summary["ear"]["median"]
        ear_std = summary["ear"]["std"]
        if ear_median < 0.05 or ear_median > 0.50 or ear_std > self.max_ear_std:
            return None
        gazes = [sample.get("gaze") for sample in samples if sample.get("gaze")]
        summary["gaze"] = max(set(gazes), key=gazes.count) if gazes else "DESCONOCIDA"
        summary["quality_median"] = float(np.median([sample["quality"] for sample in samples]))
        return summary

    def _rebuild_model(self):
        missing = [self.PROFILE_LABELS[p] for p in self.PROFILES if p not in self.profiles]
        if missing:
            self.model_ready = False
            self.thresholds = {}
            self.message = "Perfil guardado. Falta calibrar: %s" % ", ".join(missing)
            return

        open_ear = self.profiles["OPEN"]["ear"]["median"]
        drowsy_ear = self.profiles["DROWSY"]["ear"]["median"]
        asleep_ear = self.profiles["ASLEEP"]["ear"]["median"]
        if open_ear - drowsy_ear < self.min_ear_gap or drowsy_ear - asleep_ear < self.min_ear_gap:
            self.model_ready = False
            self.thresholds = {}
            self.message = "Perfiles EAR no separables: repita O, S y D con estados mas definidos"
            return

        self.thresholds = {
            "drowsy_ear_threshold": (open_ear + drowsy_ear) / 2.0,
            "ear_threshold": (drowsy_ear + asleep_ear) / 2.0,
        }
        self.model_ready = True
        self.message = "Calibracion completa: O/S/D listos"
