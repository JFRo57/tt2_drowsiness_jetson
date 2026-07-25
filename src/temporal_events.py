import time
from collections import deque


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _median(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


class EyeNormalizer(object):
    """Convierte EAR por ojo a cierre personalizado en el intervalo [0, 1]."""

    def __init__(self, config):
        self.config = config.get("fatigue", {})
        self.profile = None
        self.partial_reference = 0.5
        self.neutral_pose = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    def apply_profile(self, profile):
        self.profile = profile or None
        if not self.profile:
            return
        self.partial_reference = _clamp01(
            self.profile.get("partial_closure_normalized", 0.5)
        )
        neutral = self.profile.get("neutral_head_pose", {})
        for key in ("pitch", "yaw", "roll"):
            value = neutral.get(key)
            if value is not None:
                self.neutral_pose[key] = float(value)

    def normalize(self, metrics):
        result = {
            "left_closure_normalized": None,
            "right_closure_normalized": None,
            "closure_normalized": None,
            "partial_closure_reference": self.partial_reference,
            "valid_eye_count": 0,
            "eye_measurement_partial": False,
            "eye_measurement_valid": False,
        }
        if not self.profile:
            return result

        eyes = self.profile.get("eyes", {})
        allow_single = bool(self.config.get("allow_single_eye", True))
        fallback_reliable = bool(metrics.get("eye_reliable", False))
        closures = []
        for side, metric_key, reliability_key in (
            ("left", "left_ear", "left_eye_reliable"),
            ("right", "right_ear", "right_eye_reliable"),
        ):
            ear = metrics.get(metric_key)
            reference = eyes.get(side, {})
            reliable = bool(metrics.get(reliability_key, fallback_reliable))
            closure = self._one_eye(ear, reference) if reliable else None
            result["%s_closure_normalized" % side] = closure
            if closure is not None:
                closures.append(closure)

        result["valid_eye_count"] = len(closures)
        result["eye_measurement_partial"] = len(closures) == 1
        if len(closures) >= 2 or (allow_single and len(closures) == 1):
            result["closure_normalized"] = sum(closures) / float(len(closures))
            result["eye_measurement_valid"] = True
        return result

    @staticmethod
    def _one_eye(ear, reference):
        if ear is None:
            return None
        opened = reference.get("open")
        closed = reference.get("closed")
        if opened is None or closed is None:
            return None
        denominator = float(opened) - float(closed)
        if denominator <= 1e-6:
            return None
        return _clamp01((float(opened) - float(ear)) / denominator)

    def pose_deltas(self, metrics):
        result = {}
        for key in ("pitch", "yaw", "roll"):
            value = metrics.get(key)
            result["%s_delta" % key] = (
                float(value) - float(self.neutral_pose.get(key, 0.0))
                if value is not None
                else None
            )
        return result


class PerclosWindow(object):
    """Ventana incremental de tiempo total, valido y profundamente cerrado."""

    def __init__(self, window_seconds):
        self.window_seconds = float(window_seconds)
        self.segments = deque()
        self.total_seconds = 0.0
        self.valid_seconds = 0.0
        self.closed_seconds = 0.0
        self.last_time = None
        self.last_eligible = False
        self.last_valid = False
        self.last_closed = False

    def reset(self):
        self.segments.clear()
        self.total_seconds = 0.0
        self.valid_seconds = 0.0
        self.closed_seconds = 0.0
        self.last_time = None
        self.last_eligible = False
        self.last_valid = False
        self.last_closed = False

    def update(self, now, eligible, valid, closed):
        now = float(now)
        if self.last_time is not None and self.last_eligible:
            duration = max(0.0, now - self.last_time)
            if duration > 0.0:
                segment = (
                    self.last_time,
                    now,
                    bool(self.last_valid),
                    bool(self.last_closed and self.last_valid),
                )
                self.segments.append(segment)
                self.total_seconds += duration
                if self.last_valid:
                    self.valid_seconds += duration
                    if self.last_closed:
                        self.closed_seconds += duration
        self.last_time = now if eligible else None
        self.last_eligible = bool(eligible)
        self.last_valid = bool(valid)
        self.last_closed = bool(closed)
        self.prune(now)

    def pause(self, now):
        self.update(now, False, False, False)

    def prune(self, now):
        cutoff = float(now) - self.window_seconds
        while self.segments and self.segments[0][1] <= cutoff:
            self._remove(self.segments.popleft())
        if self.segments and self.segments[0][0] < cutoff:
            start, end, valid, closed = self.segments.popleft()
            self._remove((start, cutoff, valid, closed))
            self.segments.appendleft((cutoff, end, valid, closed))
        self.total_seconds = max(0.0, self.total_seconds)
        self.valid_seconds = max(
            0.0, min(self.valid_seconds, self.total_seconds)
        )
        self.closed_seconds = max(
            0.0, min(self.closed_seconds, self.valid_seconds)
        )

    def _remove(self, segment):
        start, end, valid, closed = segment
        duration = max(0.0, end - start)
        self.total_seconds -= duration
        if valid:
            self.valid_seconds -= duration
            if closed:
                self.closed_seconds -= duration

    def snapshot(self, now, min_coverage, min_valid_seconds):
        self.prune(now)
        coverage = (
            self.valid_seconds / self.total_seconds
            if self.total_seconds > 0.0
            else 0.0
        )
        perclos = (
            self.closed_seconds / self.valid_seconds
            if self.valid_seconds > 0.0
            else 0.0
        )
        reliable = bool(
            self.valid_seconds >= float(min_valid_seconds)
            and coverage >= float(min_coverage)
        )
        return {
            "perclos": _clamp01(perclos),
            "perclos_total_seconds": self.total_seconds,
            "perclos_valid_seconds": self.valid_seconds,
            "perclos_closed_seconds": self.closed_seconds,
            "perclos_coverage": _clamp01(coverage),
            "perclos_reliable": reliable,
        }


class TemporalEventEngine(object):
    """Transforma señales normalizadas en eventos y evidencia temporal."""

    def __init__(self, config, clock=None):
        self.config = config.get("fatigue", {})
        self.clock = clock or time.monotonic
        self.perclos_window = PerclosWindow(
            self.config.get("perclos_window_seconds", 60.0)
        )
        self.profile = None
        self.blink_baseline = {}
        self.partial_reference = 0.5
        self.blinks = deque()
        self.prolonged_blinks = deque()
        self.severe_closures = deque()
        self.yawns = deque()
        self.nods = deque()
        self.eye_state = "ABIERTO"
        self.blink_started_at = None
        self.blink_closed_at = None
        self.blink_opening_at = None
        self.blink_min_ear = None
        self.blink_min_closure = 0.0
        self.blink_invalid = False
        self.deep_duration = 0.0
        self.deep_active = False
        self.last_eye_time = None
        self.last_eye_valid = False
        self.last_deep = False
        self.eye_invalid_since = None
        self.reduced_since = None
        self.reduced_duration = 0.0
        self.yawn_state = "BOCA_NORMAL"
        self.yawn_started_at = None
        self.yawn_wide_since = None
        self.yawn_sustained = 0.0
        self.yawn_max_mar = 0.0
        self.nod_state = "CABEZA_NEUTRA"
        self.nod_started_at = None
        self.nod_down_since = None
        self.nod_max_delta = 0.0
        self.nod_max_velocity = 0.0
        self.nod_eye_overlap = False
        self.last_pitch_delta = None
        self.last_pose_time = None
        self.active_events = set()
        self.new_events = []
        self.last_frame_sequence = None

    def reset(self, clear_history=True):
        if clear_history:
            self.perclos_window.reset()
            self.blinks.clear()
            self.prolonged_blinks.clear()
            self.severe_closures.clear()
            self.yawns.clear()
            self.nods.clear()
        self.eye_state = "ABIERTO"
        self.blink_started_at = None
        self.blink_closed_at = None
        self.blink_opening_at = None
        self.blink_min_ear = None
        self.blink_min_closure = 0.0
        self.blink_invalid = False
        self.deep_duration = 0.0
        self.deep_active = False
        self.last_eye_time = None
        self.last_eye_valid = False
        self.last_deep = False
        self.eye_invalid_since = None
        self.reduced_since = None
        self.reduced_duration = 0.0
        self.yawn_state = "BOCA_NORMAL"
        self.yawn_started_at = None
        self.yawn_wide_since = None
        self.yawn_sustained = 0.0
        self.yawn_max_mar = 0.0
        self.nod_state = "CABEZA_NEUTRA"
        self.nod_started_at = None
        self.nod_down_since = None
        self.nod_max_delta = 0.0
        self.nod_max_velocity = 0.0
        self.nod_eye_overlap = False
        self.last_pitch_delta = None
        self.last_pose_time = None
        self.active_events = set()
        self.new_events = []
        self.last_frame_sequence = None

    def apply_profile(self, profile):
        self.profile = profile or None
        if not self.profile:
            return
        self.partial_reference = _clamp01(
            self.profile.get("partial_closure_normalized", 0.5)
        )
        self.blink_baseline = dict(self.profile.get("blink_baseline", {}))

    def update(self, signals, monitoring=True, paused=False):
        now = self.clock()
        self.active_events = set()
        self.new_events = []

        sequence = signals.get("frame_sequence")
        if (
            sequence is not None
            and self.last_frame_sequence is not None
            and sequence == self.last_frame_sequence
        ):
            return self.snapshot(now)
        if sequence is not None:
            self.last_frame_sequence = sequence

        if not monitoring or paused:
            self.perclos_window.pause(now)
            self._invalidate_current_events(now, reset_after=0.0)
            return self.snapshot(now)

        vision_state = signals.get("vision_state", "INICIALIZANDO")
        eye_valid = bool(signals.get("eye_measurement_valid", False))
        pose_valid = bool(signals.get("pose_reliable", True))
        valid_for_perclos = bool(
            eye_valid
            and pose_valid
            and vision_state in ("VISION_VALIDA", "VISION_DEGRADADA")
        )
        closure = signals.get("closure_normalized")
        deep_threshold = float(self.config.get("deep_closure_level", 0.80))
        deep = bool(
            valid_for_perclos
            and closure is not None
            and float(closure) >= deep_threshold
        )
        self.perclos_window.update(
            now, True, valid_for_perclos, deep
        )

        self._update_deep_closure(now, valid_for_perclos, deep)
        self._update_eye_cycle(now, signals, valid_for_perclos)
        self._update_reduced_opening(now, closure, valid_for_perclos)

        event_quality = bool(
            signals.get("face_detected", False)
            and vision_state in ("VISION_VALIDA", "VISION_DEGRADADA")
            and pose_valid
        )
        self._update_yawn(now, signals, event_quality)
        self._update_head(now, signals, event_quality)
        self._prune_events(now)
        return self.snapshot(now)

    def _update_deep_closure(self, now, valid, deep):
        if self.last_eye_time is not None and self.last_eye_valid and self.last_deep:
            self.deep_duration += max(0.0, now - self.last_eye_time)
        if valid:
            self.eye_invalid_since = None
            if not deep:
                if self.deep_duration >= float(
                    self.config.get("severe_closure_seconds", 1.2)
                ):
                    self.severe_closures.append((now, self.deep_duration))
                    self.new_events.append("CIERRE_GRAVE")
                self.deep_duration = 0.0
            self.deep_active = bool(deep)
        else:
            if self.eye_invalid_since is None:
                self.eye_invalid_since = now
            self.deep_active = False
            reset_after = float(
                self.config.get("unreliable_event_abort_seconds", 0.40)
            )
            if now - self.eye_invalid_since >= reset_after:
                self.deep_duration = 0.0
        self.last_eye_time = now
        self.last_eye_valid = bool(valid)
        self.last_deep = bool(deep)

    def _update_eye_cycle(self, now, signals, valid):
        closure = signals.get("closure_normalized")
        if not valid or closure is None:
            self._invalidate_current_events(
                now,
                float(self.config.get("unreliable_event_abort_seconds", 0.40)),
            )
            return

        start_level = float(self.config.get("blink_start_closure_level", 0.35))
        closed_level = float(self.config.get("blink_closed_closure_level", 0.75))
        reopen_level = float(self.config.get("blink_reopen_level", 0.25))
        closure = float(closure)
        current_ear = signals.get("ear")

        if self.eye_state == "ABIERTO":
            if closure >= start_level:
                self.eye_state = "CERRANDO"
                self.blink_started_at = now
                self.blink_closed_at = None
                self.blink_opening_at = None
                self.blink_min_ear = (
                    float(current_ear) if current_ear is not None else None
                )
                self.blink_min_closure = closure
                self.blink_invalid = False
        elif self.eye_state == "CERRANDO":
            self._update_blink_extremes(current_ear, closure)
            if closure >= closed_level:
                self.eye_state = "CERRADO"
                self.blink_closed_at = now
            elif closure <= reopen_level:
                self._reset_blink_cycle()
        elif self.eye_state == "CERRADO":
            self._update_blink_extremes(current_ear, closure)
            if closure < closed_level:
                self.eye_state = "ABRIENDO"
                self.blink_opening_at = now
        elif self.eye_state == "ABRIENDO":
            self._update_blink_extremes(current_ear, closure)
            if closure >= closed_level:
                self.eye_state = "CERRADO"
                self.blink_opening_at = None
            elif closure <= reopen_level:
                self._complete_blink(now)

        if self.eye_state != "ABIERTO":
            self.active_events.add("PARPADEO_EN_CURSO")
        if self.deep_active:
            self.active_events.add("CIERRE_PROFUNDO")

    def _update_blink_extremes(self, ear, closure):
        self.blink_min_closure = max(self.blink_min_closure, float(closure))
        if ear is not None:
            ear = float(ear)
            if self.blink_min_ear is None or ear < self.blink_min_ear:
                self.blink_min_ear = ear

    def _complete_blink(self, now):
        if (
            self.blink_started_at is None
            or self.blink_closed_at is None
            or self.blink_invalid
        ):
            self._reset_blink_cycle()
            return
        duration = max(0.0, now - self.blink_started_at)
        minimum = float(self.config.get("blink_min_seconds", 0.08))
        maximum_event = float(self.config.get("blink_event_max_seconds", 4.0))
        if minimum <= duration <= maximum_event:
            event = {
                "time": now,
                "duration": duration,
                "descent_seconds": max(
                    0.0, self.blink_closed_at - self.blink_started_at
                ),
                "reopen_seconds": max(
                    0.0,
                    now
                    - (
                        self.blink_opening_at
                        if self.blink_opening_at is not None
                        else self.blink_closed_at
                    ),
                ),
                "minimum_ear": self.blink_min_ear,
                "maximum_closure": self.blink_min_closure,
            }
            self.blinks.append(event)
            self.new_events.append("PARPADEO_COMPLETO")
            if duration >= self._prolonged_blink_threshold():
                self.prolonged_blinks.append(event)
                self.new_events.append("PARPADEO_PROLONGADO")
        self._reset_blink_cycle()

    def _prolonged_blink_threshold(self):
        absolute = float(
            self.config.get("prolonged_blink_absolute_seconds", 0.55)
        )
        multiplier = float(
            self.config.get("prolonged_blink_relative_multiplier", 1.8)
        )
        baseline = self.blink_baseline.get("selected", self.blink_baseline)
        reference = baseline.get("p90_seconds")
        if reference is None:
            reference = baseline.get("median_seconds")
        if reference is None:
            reference = float(self.config.get("blink_max_seconds", 0.70)) / max(
                multiplier, 1e-6
            )
        return max(absolute, float(reference) * multiplier)

    def _invalidate_current_events(self, now, reset_after):
        if self.eye_state == "ABIERTO":
            return
        self.blink_invalid = True
        if self.eye_invalid_since is None:
            self.eye_invalid_since = now
        if now - self.eye_invalid_since >= float(reset_after):
            self._reset_blink_cycle()

    def _reset_blink_cycle(self):
        self.eye_state = "ABIERTO"
        self.blink_started_at = None
        self.blink_closed_at = None
        self.blink_opening_at = None
        self.blink_min_ear = None
        self.blink_min_closure = 0.0
        self.blink_invalid = False

    def _update_reduced_opening(self, now, closure, valid):
        hysteresis = float(
            self.config.get("partial_closure_hysteresis", 0.05)
        )
        threshold = max(0.0, self.partial_reference - hysteresis)
        deep_threshold = float(self.config.get("deep_closure_level", 0.80))
        reduced = bool(
            valid
            and closure is not None
            and float(closure) >= threshold
            and float(closure) < deep_threshold
        )
        if reduced:
            if self.reduced_since is None:
                self.reduced_since = now
            self.reduced_duration = max(0.0, now - self.reduced_since)
            self.active_events.add("APERTURA_OCULAR_REDUCIDA")
            if self.reduced_duration >= float(
                self.config.get("reduced_opening_sustain_seconds", 1.5)
            ):
                self.active_events.add("APERTURA_REDUCIDA_SOSTENIDA")
        elif valid:
            self.reduced_since = None
            self.reduced_duration = 0.0

    def _update_yawn(self, now, signals, quality_valid):
        mar = signals.get("mar")
        if not quality_valid or mar is None:
            self._reset_yawn()
            return
        mar = float(mar)
        opening = float(self.config.get("mouth_open_threshold", 0.48))
        wide = float(
            self.config.get(
                "mouth_wide_threshold",
                self.config.get("mar_threshold", 0.65),
            )
        )
        closed = float(self.config.get("mouth_closed_threshold", 0.38))

        if self.yawn_state == "BOCA_NORMAL":
            if mar >= opening:
                self.yawn_state = "APERTURA_PROGRESIVA"
                self.yawn_started_at = now
                self.yawn_max_mar = mar
                self.yawn_sustained = 0.0
        elif self.yawn_state == "APERTURA_PROGRESIVA":
            self.yawn_max_mar = max(self.yawn_max_mar, mar)
            if mar >= wide:
                self.yawn_state = "APERTURA_AMPLIA_SOSTENIDA"
                self.yawn_wide_since = now
            elif mar <= closed:
                self._reset_yawn()
        elif self.yawn_state == "APERTURA_AMPLIA_SOSTENIDA":
            self.yawn_max_mar = max(self.yawn_max_mar, mar)
            if mar < wide:
                if self.yawn_wide_since is not None:
                    self.yawn_sustained += max(0.0, now - self.yawn_wide_since)
                self.yawn_wide_since = None
                self.yawn_state = "CIERRE"
            else:
                self.active_events.add("BOCA_AMPLIAMENTE_ABIERTA")
        elif self.yawn_state == "CIERRE":
            if mar >= wide:
                self.yawn_state = "APERTURA_AMPLIA_SOSTENIDA"
                self.yawn_wide_since = now
            elif mar <= closed:
                self._complete_yawn(now)

    def _complete_yawn(self, now):
        if self.yawn_started_at is None:
            self._reset_yawn()
            return
        duration = max(0.0, now - self.yawn_started_at)
        sustained = self.yawn_sustained
        if self.yawn_wide_since is not None:
            sustained += max(0.0, now - self.yawn_wide_since)
        if (
            duration >= float(self.config.get("yawn_min_seconds", 1.0))
            and sustained
            >= float(self.config.get("yawn_wide_sustain_seconds", 0.65))
        ):
            self.yawns.append(
                {
                    "time": now,
                    "duration": duration,
                    "sustained_seconds": sustained,
                    "maximum_mar": self.yawn_max_mar,
                }
            )
            self.new_events.append("BOSTEZO_COMPLETO")
        self._reset_yawn()

    def _reset_yawn(self):
        self.yawn_state = "BOCA_NORMAL"
        self.yawn_started_at = None
        self.yawn_wide_since = None
        self.yawn_sustained = 0.0
        self.yawn_max_mar = 0.0

    def _update_head(self, now, signals, quality_valid):
        pitch_delta = signals.get("pitch_delta")
        yaw_delta = signals.get("yaw_delta")
        if not quality_valid or pitch_delta is None:
            self.last_pitch_delta = None
            self.last_pose_time = None
            return

        direction = float(self.config.get("head_down_direction", 1.0))
        down_delta = float(pitch_delta) * direction
        dt = (
            max(1e-6, now - self.last_pose_time)
            if self.last_pose_time is not None
            else None
        )
        velocity = (
            (down_delta - self.last_pitch_delta) / dt
            if dt is not None and self.last_pitch_delta is not None
            else 0.0
        )
        lateral = (
            abs(float(yaw_delta))
            if yaw_delta is not None
            else 0.0
        )
        down_threshold = float(
            self.config.get("head_down_pitch_threshold", 18.0)
        )
        recover_threshold = float(
            self.config.get("head_recover_pitch_threshold", 8.0)
        )
        lateral_limit = float(
            self.config.get("head_lateral_turn_threshold", 28.0)
        )

        if lateral >= lateral_limit:
            self.active_events.add("GIRO_LATERAL_CABEZA")

        if self.nod_state == "CABEZA_NEUTRA":
            if down_delta >= down_threshold and lateral < lateral_limit:
                self.nod_state = "CABEZA_ABAJO"
                self.nod_started_at = now
                self.nod_down_since = now
                self.nod_max_delta = down_delta
                self.nod_max_velocity = max(0.0, velocity)
                self.nod_eye_overlap = self.deep_active
        elif self.nod_state == "CABEZA_ABAJO":
            self.nod_max_delta = max(self.nod_max_delta, down_delta)
            self.nod_max_velocity = max(self.nod_max_velocity, velocity)
            self.nod_eye_overlap = self.nod_eye_overlap or self.deep_active
            if down_delta <= recover_threshold:
                self._complete_nod(now)
            else:
                self.active_events.add("CABEZA_ABAJO")
                held = max(
                    0.0,
                    now
                    - (
                        self.nod_down_since
                        if self.nod_down_since is not None
                        else now
                    ),
                )
                if held >= float(
                    self.config.get("head_down_sustain_seconds", 0.8)
                ):
                    self.active_events.add("INCLINACION_CABEZA_SOSTENIDA")

        self.last_pitch_delta = down_delta
        self.last_pose_time = now

    def _complete_nod(self, now):
        duration = (
            max(0.0, now - self.nod_started_at)
            if self.nod_started_at is not None
            else 0.0
        )
        minimum_duration = float(
            self.config.get("head_nod_min_seconds", 0.35)
        )
        velocity_threshold = float(
            self.config.get("head_drop_velocity_degrees_per_second", 28.0)
        )
        if (
            duration >= minimum_duration
            and (
                self.nod_max_velocity >= velocity_threshold
                or duration
                >= float(self.config.get("head_down_sustain_seconds", 0.8))
            )
        ):
            self.nods.append(
                {
                    "time": now,
                    "duration": duration,
                    "maximum_pitch_delta": self.nod_max_delta,
                    "maximum_velocity": self.nod_max_velocity,
                    "eye_closure_overlap": self.nod_eye_overlap,
                }
            )
            self.new_events.append("CABECEO_COMPLETO")
        self.nod_state = "CABEZA_NEUTRA"
        self.nod_started_at = None
        self.nod_down_since = None
        self.nod_max_delta = 0.0
        self.nod_max_velocity = 0.0
        self.nod_eye_overlap = False

    def _prune_events(self, now):
        long_window = float(self.config.get("long_window_seconds", 60.0))
        medium_window = float(self.config.get("medium_window_seconds", 25.0))
        short_window = float(self.config.get("fast_window_seconds", 3.0))
        self._prune_dict_deque(self.blinks, now - long_window)
        self._prune_dict_deque(self.prolonged_blinks, now - medium_window)
        self._prune_tuple_deque(self.severe_closures, now - short_window)
        self._prune_dict_deque(self.yawns, now - medium_window)
        self._prune_dict_deque(self.nods, now - medium_window)

    @staticmethod
    def _prune_dict_deque(values, cutoff):
        while values and values[0]["time"] < cutoff:
            values.popleft()

    @staticmethod
    def _prune_tuple_deque(values, cutoff):
        while values and values[0][0] < cutoff:
            values.popleft()

    def snapshot(self, now=None):
        now = self.clock() if now is None else now
        perclos = self.perclos_window.snapshot(
            now,
            self.config.get("perclos_min_coverage", 0.65),
            self.config.get("perclos_min_valid_seconds", 15.0),
        )
        recent_blink_durations = [
            event["duration"] for event in self.blinks
        ]
        recent_blink_median = _median(recent_blink_durations)
        prolonged_count = len(self.prolonged_blinks)
        yawn_count = len(self.yawns)
        nod_count = len(self.nods)

        weak = []
        strong = []
        critical = []
        weak_modalities = set()
        if yawn_count == 1:
            weak.append("bostezo_aislado")
            weak_modalities.add("boca")
        if (
            "APERTURA_OCULAR_REDUCIDA" in self.active_events
            and "APERTURA_REDUCIDA_SOSTENIDA" not in self.active_events
        ):
            weak.append("apertura_ocular_reducida")
            weak_modalities.add("ojos")
        if nod_count == 1:
            weak.append("movimiento_aislado_cabeza")
            weak_modalities.add("cabeza")
        if prolonged_count == 1:
            weak.append("parpadeo_ligeramente_mayor_al_basal")
            weak_modalities.add("ojos")

        if prolonged_count >= int(
            self.config.get("prolonged_blinks_strong_count", 2)
        ):
            strong.append("parpadeos_prolongados_repetidos")
        if "APERTURA_REDUCIDA_SOSTENIDA" in self.active_events:
            strong.append("apertura_ocular_reducida_sostenida")
        if self.deep_active and self.deep_duration >= float(
            self.config.get("sustained_closure_seconds", 0.8)
        ):
            strong.append("cierre_ocular_sostenido")
        if (
            perclos["perclos_reliable"]
            and perclos["perclos"]
            >= float(self.config.get("perclos_warning_threshold", 0.25))
        ):
            strong.append("perclos_elevado")
        if nod_count >= int(self.config.get("repeated_nod_count", 2)):
            strong.append("cabeceos_repetidos")
        if "INCLINACION_CABEZA_SOSTENIDA" in self.active_events:
            strong.append("inclinacion_cabeza_sostenida")
        if yawn_count >= int(self.config.get("repeated_yawn_count", 2)) and (
            prolonged_count > 0
            or "APERTURA_REDUCIDA_SOSTENIDA" in self.active_events
        ):
            strong.append("bostezos_repetidos_con_cambio_ocular")

        critical_seconds = float(
            self.config.get("critical_closed_seconds", 2.0)
        )
        if self.deep_active and self.deep_duration >= critical_seconds:
            critical.append("cierre_ocular_critico")
        if self.deep_active and "CABEZA_ABAJO" in self.active_events and (
            self.deep_duration
            >= float(self.config.get("closure_with_head_drop_seconds", 0.8))
        ):
            critical.append("cierre_prolongado_con_caida_cabeza")
        if len(self.severe_closures) >= int(
            self.config.get("severe_closures_critical_count", 2)
        ):
            critical.append("cierres_graves_repetidos")
        if self.nods and self.nods[-1].get("eye_closure_overlap"):
            if (
                now - self.nods[-1]["time"]
                <= float(self.config.get("fast_window_seconds", 3.0))
            ):
                critical.append("cabeceo_fuerte_con_ojos_cerrados")

        events = sorted(self.active_events)
        result = {
            "active_events": events,
            "new_events": list(self.new_events),
            "eye_event_state": self.eye_state,
            "current_closure_seconds": self.deep_duration,
            "deep_closure_active": self.deep_active,
            "reduced_opening_seconds": self.reduced_duration,
            "blink_count_recent": len(self.blinks),
            "last_blink_seconds": (
                self.blinks[-1]["duration"] if self.blinks else 0.0
            ),
            "recent_blink_median_seconds": recent_blink_median,
            "baseline_blink_median_seconds": self._baseline_value(
                "median_seconds"
            ),
            "prolonged_blink_threshold_seconds": (
                self._prolonged_blink_threshold()
            ),
            "prolonged_blinks_recent": prolonged_count,
            "yawn_state": self.yawn_state,
            "recent_yawns": yawn_count,
            "nod_state": self.nod_state,
            "recent_nods": nod_count,
            "weak_evidence": weak,
            "weak_evidence_modalities": sorted(weak_modalities),
            "strong_evidence": strong,
            "critical_evidence": critical,
        }
        result.update(perclos)
        return result

    def _baseline_value(self, key):
        selected = self.blink_baseline.get("selected", self.blink_baseline)
        value = selected.get(key)
        return float(value) if value is not None else 0.0
