import cv2
import numpy as np


class PresentationUI(object):
    def __init__(self, config):
        self.full_config = config
        self.config = config["interface"]
        self.window = "Detector de somnolencia para conductores"
        self.show_landmarks = bool(self.config.get("show_landmarks", True))
        self.show_panel = bool(self.config.get("show_information_panel", True))
        self.debug_overlay = bool(self.config.get("debug_overlay", False))
        self.paused = False
        self.created = False
        self.drawn_once = False
        self.ever_visible = False
        self.closed_checks = 0
        self._panel_canvas = None
        self._panel_shape = None
        self._click_requested = False

    def create(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        camera_cfg = self.full_config.get("camera", {})
        width = int(camera_cfg.get("display_width", 1100))
        height = int(camera_cfg.get("display_height", 620))
        cv2.resizeWindow(self.window, width, height)
        cv2.setMouseCallback(self.window, self._on_mouse)
        self.created = True

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_requested = True

    def consume_click(self):
        clicked = self._click_requested
        self._click_requested = False
        return clicked

    def draw(self, frame, metrics, detector, mode_controller, gpio, capture_fps, analysis_fps, hardware_mode):
        view = self._display_frame(frame)
        if metrics.get("face_detected"):
            x1, y1, x2, y2 = metrics["rect"]
            cv2.rectangle(view, (x1, y1), (x2, y2), (40, 220, 40), 2)
            if self.show_landmarks:
                for p in metrics.get("landmarks", []):
                    cv2.circle(view, tuple(p), 1, (0, 255, 255), -1)
                for key, color in (("left_eye", (0, 255, 0)), ("right_eye", (0, 255, 0)), ("mouth", (255, 180, 0))):
                    pts = metrics.get(key)
                    if pts is not None:
                        cv2.drawContours(view, [cv2.convexHull(pts)], -1, color, 1)
        state = detector.state
        if state in ("SOMNOLENCIA", "CRITICO", "PARO_EMERGENCIA", "ERROR"):
            color = (0, 0, 255)
            scale = 0.9 if state != "CRITICO" else 1.2
            cv2.putText(view, state, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 3)
        elif state in ("SOSPECHA", "RECUPERACION"):
            cv2.putText(view, state, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        if metrics.get("calibration_active"):
            self._draw_calibration_overlay(view, metrics)
        if self.show_panel:
            view = self._add_panel(view, metrics, detector, mode_controller, gpio, capture_fps, analysis_fps, hardware_mode)
        cv2.imshow(self.window, view)
        self.drawn_once = True

    @staticmethod
    def _display_frame(frame):
        """Convert BGRx only when a three-channel image is needed by the UI."""
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame.copy()

    def _add_panel(self, frame, m, detector, mode_controller, gpio, capture_fps, analysis_fps, hardware_mode):
        panel_w = 460
        h = max(frame.shape[0], 1050 if self.debug_overlay else 430)
        shape = (h, frame.shape[1] + panel_w, 3)
        if self._panel_canvas is None or self._panel_shape != shape:
            self._panel_canvas = np.empty(shape, dtype=np.uint8)
            self._panel_shape = shape
        canvas = self._panel_canvas
        canvas.fill(0)
        canvas[:frame.shape[0], :frame.shape[1]] = frame
        x = frame.shape[1] + 15
        y = 25
        lines = [
            "Detector de somnolencia para conductores",
            "NVIDIA Jetson Nano 4 GB",
            "Camara IMX219-77IR",
            "Detector facial: %s (%s)" % (
                m.get("face_detector_backend", "--"),
                "GPU" if m.get("face_detector_accelerated") else "CPU",
            ),
            "Fallback vision: %s" % (
                m.get("face_detector_fallback", "ninguno")[:38] or "ninguno"),
            "Modo: %s (%s)" % (mode_controller.mode, mode_controller.source),
            "Estado: %s" % detector.state,
            "Vision: %s" % m.get("vision_state", "INICIALIZANDO"),
            "Motivo: %s" % detector.reason[:38],
            "Rostro detectado: %s" % ("SI" if m.get("face_detected") else "NO"),
            "Calidad: %.2f" % m.get("quality", 0.0),
            "EAR izq/der/crudo: %.3f / %.3f / %.3f" % (m.get("left_ear", 0.0), m.get("right_ear", 0.0), m.get("ear_raw", m.get("ear", 0.0))),
            "EAR filtrado: %s" % self._fmt3(m.get("ear_filtered", m.get("ear"))),
            "Cierre norm izq/der: %s / %s" % (
                self._fmt3(m.get("left_closure_normalized")),
                self._fmt3(m.get("right_closure_normalized")),
            ),
            "Cierre normalizado: %s" % self._fmt3(m.get("closure_normalized")),
            "Evento ocular: %s (%s)" % (
                m.get("eye_event_state", "--"),
                "PARCIAL" if m.get("eye_measurement_partial")
                else ("VALIDO" if m.get("eye_measurement_valid")
                      else "NO CONFIABLE"),
            ),
            "Nitidez/diferencia: %.1f / %.3f" % (m.get("eye_sharpness", 0.0), m.get("eye_ear_difference", 0.0)),
            "Umbral cerrar/abrir: %.3f / %.3f" % (
                detector.ear_threshold,
                getattr(detector, "ear_open_threshold", detector.ear_threshold),
            ),
            "Umbral apertura reducida: %s" % self._fmt3(detector.drowsy_ear_threshold),
            "Perfiles: %s" % m.get("calibration_profiles", "O:-- R:-- C:--"),
            "Perfil detectado: %s %.0f%% / %.1f s" % (
                m.get("calibrated_profile", "NO_CALIBRADO"),
                float(m.get("calibrated_profile_confidence", 0.0)) * 100.0,
                float(m.get("calibrated_profile_seconds", 0.0)),
            ),
            "Ojos cerrados estables: %.2f s" % m.get("closed_seconds", 0.0),
            "Parpadeos recientes: %s" % m.get("blink_count_recent", 0),
            "Basal parpadeo: %.3f s" % m.get("baseline_blink_median_seconds", 0.0),
            "PERCLOS/cobertura: %.1f / %.1f %% (%s)" % (
                m.get("perclos", 0.0) * 100.0,
                m.get("perclos_coverage", 0.0) * 100.0,
                "OK" if m.get("perclos_reliable") else "PARCIAL",
            ),
            "MAR: %.3f  Bostezo: %s" % (m.get("mar", 0.0), "SI" if m.get("possible_yawn") else "NO"),
            "Pitch/Yaw/Roll: %s / %s / %s" % (self._fmt(m.get("pitch")), self._fmt(m.get("yaw")), self._fmt(m.get("roll"))),
            "Cabeceo posible: %s" % ("SI" if m.get("possible_nod") else "NO"),
            "Cabeceos/bostezos ventana: %d / %d" % (
                m.get("recent_nods", 0), m.get("recent_yawns", 0)),
            "Mirada: %s" % m.get("gaze", "DESCONOCIDA"),
            "Sin rostro: %.1f s" % m.get("no_face_seconds", 0.0),
            "Luminosidad media: %.1f" % m.get("brightness", 0.0),
            "Buzzer: %s" % (self._buzzer_label(gpio)),
            "Tiempo ejecucion: %s" % self._fmt_duration(m.get("runtime_seconds", 0.0)),
            "FPS captura/analisis/objetivo: %.1f / %.1f / %.1f" % (
                capture_fps,
                analysis_fps,
                m.get("target_analysis_fps", 0.0),
            ),
            "Tiempo analisis/frame: %.1f ms" % m.get("analysis_ms", 0.0),
            "Antiguedad del cuadro: %.1f ms" % m.get("frame_age_ms", 0.0),
            "Ruta de vision: %s" % m.get("analysis_source", "--"),
            "Etapas pre/loc/lm/feat: %s" % self._analysis_stages(m),
            "Saltados analisis/captura: %d / %d" % (
                m.get("analysis_skipped_frames", 0),
                m.get("capture_dropped_frames", 0),
            ),
            "Velocidad deteccion: %s" % self._speed_label(analysis_fps),
            "Hardware: %s" % hardware_mode,
            "GPIO error: %s" % (gpio.error[:36] if gpio.error else "ninguno"),
            "Teclas: 1 Auto  2 Mant  3 Paro",
            "Prueba: 4 Alerta 5 Sospecha 6 Somnol.",
            "7 Critico 8 Recuperacion 0 Real",
            "Calibrar: C Completa O Abiertos",
            "S Apertura reducida D Cerrados B Blink",
            "L Landmarks I Panel V Debug M Mute P Pausa",
            "R Reset  Q Salir",
        ]
        if not self.debug_overlay:
            lines = [
                lines[0], lines[3], lines[5], lines[6], lines[7], lines[8],
                "Cierre normalizado: %s" % self._fmt3(m.get("closure_normalized")),
                "PERCLOS/cobertura: %.1f / %.1f %%" % (
                    m.get("perclos", 0.0) * 100.0,
                    m.get("perclos_coverage", 0.0) * 100.0,
                ),
                "Alerta activa: %s" % m.get("active_alert", "--"),
                "FPS captura/analisis: %.1f / %.1f" % (capture_fps, analysis_fps),
                "C Calibracion completa  V Debug  Q Salir",
            ]
        for line in lines:
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            y += 22
        self._led(canvas, x + 30, h - 45, (0, 220, 0), gpio.leds.get("green"), "VERDE")
        self._led(canvas, x + 145, h - 45, (0, 220, 220), gpio.leds.get("yellow"), "AMARILLO")
        self._led(canvas, x + 290, h - 45, (0, 0, 255), gpio.leds.get("red"), "ROJO")
        return canvas


    @staticmethod
    def _draw_calibration_overlay(view, metrics):
        labels = {
            "OPEN": "OJOS ABIERTOS",
            "REDUCED": "APERTURA OCULAR REDUCIDA",
            "CLOSED": "OJOS COMPLETAMENTE CERRADOS",
            "DYNAMIC_NATURAL": "OBSERVACION NATURAL DE PARPADEOS",
            "DYNAMIC_VOLUNTARY": "PARPADEOS VOLUNTARIOS NORMALES",
        }
        target = metrics.get("calibration_target", "")
        progress = max(0.0, min(1.0, float(metrics.get("calibration_progress", 0.0))))
        preparing = bool(metrics.get("calibration_preparing", False))
        waiting = bool(metrics.get("calibration_waiting_confirmation", False))
        remaining = float(metrics.get("calibration_prepare_remaining_seconds", 0.0))
        mode = "CONFIRMAR" if waiting else ("PREPARANDO" if preparing else "CALIBRANDO")
        message = metrics.get("calibration_message") or "Vehiculo detenido; no mueva la cabeza"
        if len(message) > 92:
            message = message[:89] + "..."
        samples = int(metrics.get("calibration_valid_samples", 0) or 0)
        attempts = int(metrics.get("calibration_sample_attempts", 0) or 0)
        width = max(120, view.shape[1] - 80)
        cv2.rectangle(view, (25, 70), (view.shape[1] - 25, 174), (15, 15, 15), -1)
        cv2.putText(view, "%s: %s" % (mode, labels.get(target, target)), (40, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        cv2.putText(view, message, (40, 124),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)
        if waiting:
            detail = "Click izquierdo, ESPACIO o ENTER para continuar"
        elif preparing:
            detail = "Comienza en %.1f s" % remaining
        else:
            detail = "Muestras validas: %d / intentos: %d" % (samples, attempts)
        cv2.putText(view, detail, (40, 148), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (220, 220, 220), 1)
        cv2.rectangle(view, (40, 158), (40 + width, 169), (90, 90, 90), 1)
        cv2.rectangle(view, (40, 158), (40 + int(width * progress), 169), (0, 220, 255), -1)

    @staticmethod
    def _buzzer_label(gpio):
        if not gpio.buzzer_on:
            return "apagado"
        frequency = getattr(gpio, "buzzer_frequency", 0)
        if frequency:
            return "ENCENDIDO %d Hz" % frequency
        return "ENCENDIDO"

    @staticmethod
    def _fmt(value):
        return "--" if value is None else "%.1f" % value

    @staticmethod
    def _fmt3(value):
        return "--" if value is None else "%.3f" % value

    @staticmethod
    def _analysis_stages(metrics):
        stages = metrics.get("analysis_breakdown_ms", {})
        return "%.1f/%.1f/%.1f/%.1f ms" % (
            stages.get("preprocess", 0.0),
            stages.get("locate", 0.0),
            stages.get("landmarks", 0.0),
            stages.get("features", 0.0),
        )

    @staticmethod
    def _fmt_duration(seconds):
        seconds = int(max(0.0, float(seconds or 0.0)))
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return "%d:%02d:%02d" % (hours, minutes, sec)
        return "%02d:%02d" % (minutes, sec)

    @staticmethod
    def _speed_label(analysis_fps):
        if analysis_fps >= 20.0:
            return "RAPIDA"
        if analysis_fps >= 12.0:
            return "ACEPTABLE"
        return "LENTA"

    @staticmethod
    def _led(canvas, x, y, color, on, label):
        c = color if on else tuple(int(v * 0.25) for v in color)
        cv2.circle(canvas, (x, y), 14, c, -1)
        cv2.circle(canvas, (x, y), 14, (180, 180, 180), 1)
        cv2.putText(canvas, label, (x - 28, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    def handle_key(self, key, app):
        if key in (ord("q"), ord("Q"), 27):
            app.shutdown.request("Tecla de salida")
        elif key == ord("1"):
            app.gpio.set_simulated_mode("AUTOMATIC")
        elif key == ord("2"):
            app.gpio.set_simulated_mode("MAINTENANCE")
        elif key == ord("3"):
            app.gpio.set_simulated_mode("EMERGENCY")
        elif key == ord("4"):
            app.set_forced_test_state("ALERTA")
        elif key == ord("5"):
            app.set_forced_test_state("SOSPECHA")
        elif key == ord("6"):
            app.set_forced_test_state("SOMNOLENCIA")
        elif key == ord("7"):
            app.set_forced_test_state("CRITICO")
        elif key == ord("8"):
            app.set_forced_test_state("RECUPERACION")
        elif key == ord("0"):
            app.set_forced_test_state(None)
        elif key in (ord("c"), ord("C")):
            app.start_calibration("FULL")
        elif key in (ord("o"), ord("O")):
            app.start_calibration("OPEN")
        elif key in (ord("s"), ord("S")):
            app.start_calibration("REDUCED")
        elif key in (ord("d"), ord("D")):
            app.start_calibration("CLOSED")
        elif key in (ord("b"), ord("B")):
            app.start_calibration("DYNAMIC")
        elif key in (32, 10, 13):
            app.confirm_calibration_step()
        elif key in (ord("l"), ord("L")):
            self.show_landmarks = not self.show_landmarks
        elif key in (ord("i"), ord("I")):
            self.show_panel = not self.show_panel
        elif key in (ord("v"), ord("V")):
            self.debug_overlay = not self.debug_overlay
        elif key in (ord("m"), ord("M")):
            app.config["buzzer"]["muted"] = not app.config["buzzer"].get("muted", False)
            app.gpio.buzzer_cfg["muted"] = app.config["buzzer"]["muted"]
        elif key in (ord("p"), ord("P")):
            self.paused = not self.paused
        elif key in (ord("r"), ord("R")):
            app.forced_test_state = None
            app.detector.reset()

    def is_closed(self):
        if not self.created or not self.drawn_once:
            return False
        try:
            visible = cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE)
        except Exception:
            visible = 0
        if visible > 0:
            self.ever_visible = True
            self.closed_checks = 0
        elif visible == 0 or self.ever_visible:
            self.closed_checks += 1
        return self.closed_checks >= 3

    def close(self):
        cv2.destroyAllWindows()
