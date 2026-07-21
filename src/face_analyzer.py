import math
import os
from collections import deque

import cv2
import dlib
import numpy as np


LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))
MOUTH = list(range(48, 68))


def shape_to_np(shape):
    pts = np.zeros((68, 2), dtype="int")
    for i in range(68):
        p = shape.part(i)
        pts[i] = (p.x, p.y)
    return pts


def rect_area(rect):
    return max(0, rect.right() - rect.left()) * max(0, rect.bottom() - rect.top())


class FaceAnalyzer(object):
    def __init__(self, config):
        self.config = config
        dcfg = config["dlib"]
        predictor_path = dcfg["predictor_path"]
        if not os.path.isabs(predictor_path):
            predictor_path = os.path.abspath(predictor_path)
        if not os.path.exists(predictor_path):
            raise RuntimeError("Predictor facial no encontrado: %s" % predictor_path)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.tracker = None
        self.last_rect = None
        self.frame_index = 0
        self.gaze_history = deque(maxlen=7)
        self.angle_history = deque(maxlen=5)
        self.clahe = cv2.createCLAHE(
            clipLimit=float(config["preprocessing"]["clahe_clip_limit"]),
            tileGridSize=(int(config["preprocessing"]["clahe_grid_size"]), int(config["preprocessing"]["clahe_grid_size"])),
        )

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        pcfg = self.config["preprocessing"]
        if pcfg.get("use_clahe", True):
            gray = self.clahe.apply(gray)
        if pcfg.get("use_gamma", False):
            gamma = max(0.1, float(pcfg.get("gamma", 1.0)))
            table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
            gray = cv2.LUT(gray, table)
        return gray, brightness

    def analyze(self, frame):
        gray, brightness = self.preprocess(frame)
        self.frame_index += 1
        rect = self._locate_face(gray)
        if rect is None:
            return {"face_detected": False, "brightness": brightness, "quality": 0.0}
        shape = self.predictor(gray, rect)
        landmarks = shape_to_np(shape)
        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]
        mouth = landmarks[MOUTH]
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = self.mouth_aspect_ratio(mouth)
        pitch, yaw, roll = self.estimate_head_pose(landmarks, frame.shape)
        gaze = self.estimate_gaze(gray, landmarks)
        quality = self.quality_score(rect, landmarks, gray.shape)
        return {
            "face_detected": True,
            "rect": (rect.left(), rect.top(), rect.right(), rect.bottom()),
            "landmarks": landmarks,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "mouth": mouth,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "ear": ear,
            "mar": mar,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "gaze": gaze,
            "quality": quality,
            "brightness": brightness,
        }

    def _locate_face(self, gray):
        interval = max(1, int(self.config["dlib"]["detection_interval_frames"]))
        use_tracker = bool(self.config["dlib"].get("use_correlation_tracker", True))
        need_detect = self.last_rect is None or self.frame_index % interval == 0
        if use_tracker and self.tracker is not None and not need_detect:
            quality = self.tracker.update(gray)
            if quality >= 6.5:
                pos = self.tracker.get_position()
                self.last_rect = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
                return self.last_rect
            need_detect = True
        if need_detect:
            rects = self.detector(gray, int(self.config["dlib"].get("upsample", 0)))
            if not rects:
                self.tracker = None
                self.last_rect = None
                return None
            rect = max(rects, key=rect_area)
            self.last_rect = rect
            if use_tracker:
                self.tracker = dlib.correlation_tracker()
                self.tracker.start_track(gray, rect)
            return rect
        return self.last_rect

    @staticmethod
    def eye_aspect_ratio(eye):
        a = np.linalg.norm(eye[1] - eye[5])
        b = np.linalg.norm(eye[2] - eye[4])
        c = np.linalg.norm(eye[0] - eye[3])
        return float((a + b) / (2.0 * c)) if c > 1e-6 else 0.0

    @staticmethod
    def mouth_aspect_ratio(mouth):
        a = np.linalg.norm(mouth[13] - mouth[19])
        b = np.linalg.norm(mouth[14] - mouth[18])
        c = np.linalg.norm(mouth[15] - mouth[17])
        d = np.linalg.norm(mouth[12] - mouth[16])
        return float((a + b + c) / (3.0 * d)) if d > 1e-6 else 0.0

    def estimate_head_pose(self, pts, frame_shape):
        image_points = np.array([pts[30], pts[8], pts[36], pts[45], pts[48], pts[54]], dtype="double")
        model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])
        h, w = frame_shape[:2]
        focal = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype="double")
        dist = np.zeros((4, 1))
        try:
            ok, rvec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None, None, None
            rot, _ = cv2.Rodrigues(rvec)
            proj = np.hstack((rot, np.zeros((3, 1))))
            angles = cv2.decomposeProjectionMatrix(proj)[6]
            pitch, yaw, roll = [float(a) for a in angles.ravel()]
        except Exception:
            return None, None, None
        self.angle_history.append((pitch, yaw, roll))
        arr = np.array(self.angle_history)
        return tuple(np.median(arr, axis=0))

    def estimate_gaze(self, gray, pts):
        labels = []
        for idxs in (LEFT_EYE, RIGHT_EYE):
            poly = pts[idxs]
            x, y, w, h = cv2.boundingRect(poly)
            if w < 8 or h < 4:
                continue
            pad = 2
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            m = cv2.moments(th)
            if m["m00"] <= 1:
                continue
            cx = (m["m10"] / m["m00"]) / max(1.0, float(x1 - x0))
            cy = (m["m01"] / m["m00"]) / max(1.0, float(y1 - y0))
            if cx < 0.35:
                labels.append("IZQUIERDA")
            elif cx > 0.65:
                labels.append("DERECHA")
            elif cy < 0.30:
                labels.append("ARRIBA")
            elif cy > 0.75:
                labels.append("ABAJO")
            else:
                labels.append("CENTRO")
        label = max(set(labels), key=labels.count) if labels else "DESCONOCIDA"
        self.gaze_history.append(label)
        return max(set(self.gaze_history), key=list(self.gaze_history).count) if self.gaze_history else label

    @staticmethod
    def quality_score(rect, pts, shape):
        h, w = shape[:2]
        inside = rect.left() >= 0 and rect.top() >= 0 and rect.right() < w and rect.bottom() < h
        area = float(rect_area(rect)) / float(max(1, w * h))
        eye_span = np.linalg.norm(pts[36] - pts[45])
        score = min(1.0, area * 8.0) * (1.0 if inside else 0.6)
        if eye_span < 25:
            score *= 0.5
        return float(score)
