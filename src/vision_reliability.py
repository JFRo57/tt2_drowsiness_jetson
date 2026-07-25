import time


class VisionReliabilityMonitor(object):
    """Clasifica la disponibilidad de la medicion, no la fatiga."""

    INITIALIZING = "INICIALIZANDO"
    VALID = "VISION_VALIDA"
    DEGRADED = "VISION_DEGRADADA"
    FACE_NOT_VISIBLE = "ROSTRO_NO_VISIBLE"
    CAMERA_FAILURE = "CAMARA_OBSTRUIDA_O_FALLO"

    STATES = (
        INITIALIZING,
        VALID,
        DEGRADED,
        FACE_NOT_VISIBLE,
        CAMERA_FAILURE,
    )

    def __init__(self, config, clock=None):
        self.config = config.get("vision_reliability", {})
        self.clock = clock or time.monotonic
        self.state = self.INITIALIZING
        self.reason = "Esperando mediciones validas"
        self.started_at = self.clock()
        self.valid_since = None
        self.face_missing_since = None
        self.frame_missing_since = None
        self.low_fps_since = None
        self.last_valid_at = None
        self.valid_eye_count = 0
        self.measurement_partial = False

    def reset(self):
        now = self.clock()
        self.state = self.INITIALIZING
        self.reason = "Esperando mediciones validas"
        self.started_at = now
        self.valid_since = None
        self.face_missing_since = None
        self.frame_missing_since = None
        self.low_fps_since = None
        self.last_valid_at = None
        self.valid_eye_count = 0
        self.measurement_partial = False

    def update(self, metrics, analysis_fps=0.0, camera_error=None):
        now = self.clock()
        self.valid_eye_count = 0
        self.measurement_partial = False

        if camera_error:
            self.state = self.CAMERA_FAILURE
            self.reason = "Fallo de camara: %s" % camera_error
            return self.snapshot(now)

        frame_available = bool(metrics.get("frame_available", True))
        if not frame_available:
            if self.frame_missing_since is None:
                self.frame_missing_since = now
            missing = now - self.frame_missing_since
            failure_after = float(
                self.config.get("camera_failure_seconds", 6.0)
            )
            if missing >= failure_after:
                self.state = self.CAMERA_FAILURE
                self.reason = "La camara no entrega cuadros por %.1f s" % missing
            else:
                self.state = self.DEGRADED
                self.reason = "Flujo de camara intermitente por %.1f s" % missing
            self.valid_since = None
            return self.snapshot(now)
        self.frame_missing_since = None

        if not metrics.get("face_detected", False):
            if self.face_missing_since is None:
                self.face_missing_since = now
            missing = now - self.face_missing_since
            no_face_after = float(self.config.get("no_face_seconds", 2.0))
            failure_after = float(
                self.config.get("persistent_loss_seconds", 8.0)
            )
            if missing >= failure_after:
                self.state = self.CAMERA_FAILURE
                self.reason = (
                    "Supervision visual perdida por %.1f s; no se puede "
                    "distinguir obstruccion de ausencia del conductor"
                ) % missing
            elif missing >= no_face_after:
                self.state = self.FACE_NOT_VISIBLE
                self.reason = "Rostro no visible por %.1f s" % missing
            else:
                self.state = self.DEGRADED
                self.reason = "Perdida breve del rostro por %.1f s" % missing
            self.valid_since = None
            return self.snapshot(now)
        self.face_missing_since = None

        quality = float(metrics.get("quality", 0.0))
        minimum_quality = float(self.config.get("min_quality", 0.25))
        left_reliable = bool(
            metrics.get("left_eye_reliable", metrics.get("eye_reliable", False))
        )
        right_reliable = bool(
            metrics.get("right_eye_reliable", metrics.get("eye_reliable", False))
        )
        self.valid_eye_count = int(left_reliable) + int(right_reliable)
        self.measurement_partial = self.valid_eye_count == 1

        brightness = float(metrics.get("brightness", 0.0))
        light_ok = (
            brightness >= float(self.config.get("min_brightness", 12.0))
            and brightness <= float(self.config.get("max_brightness", 250.0))
        )
        pose_ok = bool(metrics.get("pose_reliable", True))

        minimum_fps = float(self.config.get("min_analysis_fps", 8.0))
        fps_grace = float(self.config.get("low_fps_grace_seconds", 3.0))
        startup_grace = float(self.config.get("startup_fps_grace_seconds", 3.0))
        fps_is_known = analysis_fps is not None and float(analysis_fps) > 0.0
        low_fps = fps_is_known and float(analysis_fps) < minimum_fps
        if low_fps:
            if self.low_fps_since is None:
                self.low_fps_since = now
        else:
            self.low_fps_since = None
        fps_invalid = bool(
            self.low_fps_since is not None
            and now - self.low_fps_since >= fps_grace
            and now - self.started_at >= startup_grace
        )

        degraded_reasons = []
        if quality < minimum_quality:
            degraded_reasons.append("landmarks de baja calidad")
        if self.valid_eye_count == 0:
            degraded_reasons.append("ningun ojo es medible")
        elif self.valid_eye_count == 1:
            degraded_reasons.append("solo un ojo es medible")
        if not pose_ok:
            degraded_reasons.append("pose invalida el EAR")
        if not light_ok:
            degraded_reasons.append("iluminacion fuera de rango")
        if fps_invalid:
            degraded_reasons.append(
                "FPS %.1f inferior al minimo" % float(analysis_fps)
            )

        fully_valid = (
            quality >= minimum_quality
            and self.valid_eye_count >= 2
            and pose_ok
            and light_ok
            and not fps_invalid
        )
        if fully_valid:
            if self.valid_since is None:
                self.valid_since = now
            self.last_valid_at = now
            initialize_seconds = float(
                self.config.get("initial_valid_seconds", 0.5)
            )
            if now - self.valid_since < initialize_seconds:
                self.state = self.INITIALIZING
                self.reason = "Confirmando vision estable"
            else:
                self.state = self.VALID
                self.reason = "Rostro, ojos, pose e iluminacion validos"
        else:
            self.valid_since = None
            self.state = self.DEGRADED
            self.reason = (
                "Vision parcial: %s"
                % ", ".join(degraded_reasons or ["calidad no confirmada"])
            )
        return self.snapshot(now)

    def snapshot(self, now=None):
        now = self.clock() if now is None else now
        return {
            "state": self.state,
            "reason": self.reason,
            "valid_eye_count": self.valid_eye_count,
            "measurement_partial": self.measurement_partial,
            "face_missing_seconds": (
                max(0.0, now - self.face_missing_since)
                if self.face_missing_since is not None
                else 0.0
            ),
            "frame_missing_seconds": (
                max(0.0, now - self.frame_missing_since)
                if self.frame_missing_since is not None
                else 0.0
            ),
        }
