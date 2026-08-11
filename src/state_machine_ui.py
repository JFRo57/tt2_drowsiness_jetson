import cv2
import numpy as np


class StateMachineUI(object):
    """Ventana auxiliar para observar decisiones y tiempos del detector."""

    WINDOW = "Maquina de estados - Somnolencia"
    COLORS = {
        "ALERTA": (70, 190, 70), "SOSPECHA": (0, 210, 255),
        "SOMNOLENCIA": (0, 125, 255), "CRITICO": (45, 45, 235),
        "RECUPERACION": (220, 150, 40),
    }

    def __init__(self, config):
        self.config = config
        self.enabled = bool(config.get("interface", {}).get(
            "state_machine_enabled", False))
        self.created = False
        self.drawn_once = False
        self.ever_visible = False
        self.closed_checks = 0
        self.canvas = np.zeros((680, 980, 3), dtype=np.uint8)

    def create(self):
        if self.enabled:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW, 980, 680)
            self.created = True

    def draw(self, metrics, detector):
        if self.enabled:
            cv2.imshow(self.WINDOW, self.render(metrics, detector))
            self.drawn_once = True

    def render(self, m, detector):
        c = self.canvas
        c[:] = (18, 21, 27)
        current = str(getattr(detector, "state", "--"))
        previous = str(getattr(detector, "previous_state", "--") or "--")
        self._text(c, "MAQUINA DE ESTADOS EN TIEMPO REAL", 28, 38, .78,
                   (245, 245, 245), 2)
        self._text(c, "Transicion: %s  ->  %s" % (previous, current),
                   30, 72, .55, (185, 195, 210))
        nodes = {"ALERTA": (105, 165), "SOSPECHA": (300, 165),
                 "SOMNOLENCIA": (520, 165), "CRITICO": (745, 165),
                 "RECUPERACION": (745, 300)}
        self._arrow(c, (177, 165), (228, 165), "evidencia")
        self._arrow(c, (372, 165), (448, 165), "4 s")
        self._arrow(c, (592, 165), (673, 165), "critica")
        self._arrow(c, (745, 203), (745, 260), "abre")
        self._arrow(c, (673, 300), (375, 190), "estable")
        self._arrow(c, (300, 203), (680, 278), "cierre >= 2 s")
        for state, center in nodes.items():
            self._node(c, center, state, state == current, state == previous)

        self._text(c, "Decision actual", 30, 380, .62, (240, 240, 240), 2)
        reason = str(getattr(detector, "reason", "--"))
        self._text(c, reason[:83], 30, 412, .45, (210, 220, 230))
        if len(reason) > 83:
            self._text(c, reason[83:166], 30, 434, .45, (210, 220, 230))

        fatigue = self.config.get("fatigue", {})
        closed = float(m.get("current_closure_seconds", 0.0) or 0.0)
        closure = m.get("closure_normalized")
        rows = [
            ("Vision", m.get("vision_state", "--")),
            ("Medicion ocular", "VALIDA" if m.get("eye_measurement_valid") else "NO VALIDA"),
            ("Perfil ocular instantaneo", m.get("calibrated_profile", "--")),
            ("Cierre normalizado", "--" if closure is None else "%.3f" % closure),
            ("Cierre profundo", ("PAUSA BREVE" if m.get("deep_closure_dropout_active")
                                  else ("SI" if m.get("deep_closure_active") else "NO"))),
            ("Tiempo de cierre", "%.2f s" % closed),
            ("Evidencia debil", self._items(m.get("weak_evidence"))),
            ("Evidencia fuerte", self._items(m.get("strong_evidence"))),
            ("Evidencia critica", self._items(m.get("critical_evidence"))),
        ]
        y = 478
        for label, value in rows:
            self._text(c, label + ":", 30, y, .45, (150, 165, 185))
            self._text(c, value, 205, y, .45, (235, 235, 235))
            y += 23
        elapsed = self._elapsed(getattr(detector, "strong_since", None),
                                m.get("runtime_monotonic"))
        self._bar(c, 570, 405, "Cierre sostenido", closed,
                  float(fatigue.get("sustained_closure_seconds", .8)))
        self._bar(c, 570, 475, "Cierre critico", closed,
                  float(fatigue.get("critical_closed_seconds", 2.0)))
        self._bar(c, 570, 545, "Confirmacion somnolencia", elapsed,
                  float(fatigue.get("suspicion_to_somnolence_seconds", 4.0)))
        return c

    @staticmethod
    def _elapsed(started, now):
        return 0.0 if started is None or now is None else max(0.0, now - started)

    @staticmethod
    def _items(values):
        return ", ".join(list(values or [])[:2]) or "ninguna"

    def _node(self, c, center, label, active, previous):
        color = self.COLORS[label]
        cv2.ellipse(c, center, (72, 37), 0, 0, 360,
                    color if active else (35, 40, 49), -1 if active else 2,
                    cv2.LINE_AA)
        if active or previous:
            cv2.ellipse(c, center, (78, 43), 0, 0, 360,
                        (245, 245, 245) if active else (125, 135, 150),
                        2 if active else 1, cv2.LINE_AA)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .52, 1)[0]
        self._text(c, label, center[0] - size[0] // 2, center[1] + 6,
                   .52, (15, 18, 22) if active else color)

    def _arrow(self, c, start, end, label):
        cv2.arrowedLine(c, start, end, (105, 115, 130), 1, cv2.LINE_AA,
                        tipLength=.12)
        self._text(c, label, (start[0] + end[0]) // 2 - 20,
                   (start[1] + end[1]) // 2 - 7, .34, (125, 135, 150))

    def _bar(self, c, x, y, label, value, target):
        ratio = min(1.0, value / max(.001, target))
        self._text(c, "%s: %.2f / %.2f s" % (label, value, target),
                   x, y, .45, (215, 220, 230))
        cv2.rectangle(c, (x, y + 12), (x + 365, y + 31), (65, 70, 80), 1)
        cv2.rectangle(c, (x + 2, y + 14),
                      (x + 2 + int(361 * ratio), y + 29), (0, 190, 245), -1)

    @staticmethod
    def _text(c, text, x, y, scale, color, thickness=1):
        cv2.putText(c, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)

    def is_closed(self):
        if not self.enabled or not self.created or not self.drawn_once:
            return False
        try:
            visible = cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE)
        except Exception:
            visible = 0
        if visible > 0:
            self.ever_visible, self.closed_checks = True, 0
        elif visible == 0 or self.ever_visible:
            self.closed_checks += 1
        return self.closed_checks >= 3

    def disable(self):
        self.enabled = False
        try:
            cv2.destroyWindow(self.WINDOW)
        except Exception:
            pass
