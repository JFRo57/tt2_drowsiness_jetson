import math
import os
import time
from collections import deque

import cv2
import dlib
import numpy as np

from .face_detector import FaceDetectorBackend


LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))
MOUTH = list(range(48, 68))


def shape_to_np(shape):
    pts = np.empty((68, 2), dtype=np.int32)
    for i in range(68):
        point = shape.part(i)
        pts[i, 0] = point.x
        pts[i, 1] = point.y
    return pts


def rect_area(rect):
    return max(0, rect.right() - rect.left()) * max(0, rect.bottom() - rect.top())


def point_distance(a, b):
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


class FaceAnalyzer(object):
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ], dtype=np.float64)

    def __init__(self, config):
        self.config = config
        dcfg = config["dlib"]
        pcfg = config["preprocessing"]
        predictor_path = dcfg["predictor_path"]
        if not os.path.isabs(predictor_path):
            predictor_path = os.path.abspath(predictor_path)
        if not os.path.exists(predictor_path):
            raise RuntimeError("Predictor facial no encontrado: %s" % predictor_path)

        self.face_detector = FaceDetectorBackend(config)
        self.predictor = dlib.shape_predictor(predictor_path)
        self.tracker = None
        self.last_rect = None
        self.frame_index = 0
        self.last_detection_source = "inicial"
        self.detection_interval = max(1, int(dcfg.get("detection_interval_frames", 12)))
        self.no_face_detection_interval = max(1, int(dcfg.get("no_face_detection_interval_frames", 3)))
        self.detector_scale = max(0.25, min(1.0, float(dcfg.get("detector_scale", 0.5))))
        default_tracking = "correlation" if dcfg.get("use_correlation_tracker", False) else "landmarks"
        self.tracking_mode = str(dcfg.get("tracking_mode", default_tracking)).lower()
        self.tracker_quality_threshold = float(dcfg.get("tracker_quality_threshold", 6.5))
        self.pose_interval = max(1, int(dcfg.get("pose_interval_frames", 2)))
        self.gaze_interval = max(1, int(dcfg.get("gaze_interval_frames", 2)))

        # Estabilizacion temporal para una camara instalada en un vehiculo. La
        # traslacion global sigue al rostro con rapidez, mientras la forma de
        # los landmarks y el tamano del ROI se suavizan con mas fuerza.
        scfg = config.get("stability", {})
        self.landmark_shape_alpha = self._clamp01(
            scfg.get("landmark_shape_alpha", 0.30)
        )
        self.landmark_translation_alpha = self._clamp01(
            scfg.get("landmark_translation_alpha", 0.75)
        )
        self.tracking_rect_alpha = self._clamp01(
            scfg.get("tracking_rect_alpha", 0.35)
        )
        self.redetection_rect_alpha = self._clamp01(
            scfg.get("redetection_rect_alpha", 0.30)
        )
        self.redetection_min_iou = max(
            0.0, min(1.0, float(scfg.get("redetection_min_iou", 0.15)))
        )
        self.redetection_max_center_shift = max(
            0.05, float(scfg.get("redetection_max_center_shift", 0.45))
        )
        self.redetection_miss_tolerance = max(
            0, int(scfg.get("redetection_miss_tolerance", 2))
        )
        self.min_eye_width_pixels = max(
            4.0, float(scfg.get("min_eye_width_pixels", 10.0))
        )
        self.min_eye_sharpness = max(
            0.0, float(scfg.get("min_eye_sharpness", 12.0))
        )
        self.max_eye_ear_difference = max(
            0.01, float(scfg.get("max_eye_ear_difference", 0.12))
        )
        self.max_eye_ear_difference_ratio = max(
            0.1, float(scfg.get("max_eye_ear_difference_ratio", 0.65))
        )
        self.max_eye_yaw_degrees = max(
            5.0, float(scfg.get("max_eye_yaw_degrees", 32.0))
        )
        ear_window = max(1, int(scfg.get("ear_median_window", 3)))
        if ear_window % 2 == 0:
            ear_window += 1
        self.ear_history = deque(maxlen=ear_window)
        self.filtered_landmarks = None
        self.redetection_miss_count = 0

        self.gaze_history = deque(maxlen=7)
        self.angle_history = deque(maxlen=5)
        self.last_pose = (None, None, None)
        self.last_gaze = "DESCONOCIDA"
        self.camera_shape = None
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        self.clahe = cv2.createCLAHE(
            clipLimit=float(pcfg["clahe_clip_limit"]),
            tileGridSize=(int(pcfg["clahe_grid_size"]), int(pcfg["clahe_grid_size"])),
        )
        self.use_clahe = bool(pcfg.get("use_clahe", True))
        self.adaptive_clahe = bool(pcfg.get("adaptive_clahe", True))
        self.clahe_dark_threshold = float(pcfg.get("clahe_dark_threshold", 75.0))
        self.clahe_bright_threshold = float(pcfg.get("clahe_bright_threshold", 205.0))
        self.gamma_table = None
        if pcfg.get("use_gamma", False):
            gamma = max(0.1, float(pcfg.get("gamma", 1.0)))
            self.gamma_table = np.array(
                [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
                dtype=np.uint8,
            )

    def preprocess(self, frame):
        if frame.ndim == 3 and frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(cv2.mean(gray)[0])
        needs_clahe = (
            self.use_clahe
            and (
                not self.adaptive_clahe
                or brightness < self.clahe_dark_threshold
                or brightness > self.clahe_bright_threshold
            )
        )
        if needs_clahe:
            gray = self.clahe.apply(gray)
        if self.gamma_table is not None:
            gray = cv2.LUT(gray, self.gamma_table)
        return gray, brightness

    def analyze(self, frame):
        total_started = time.perf_counter()
        gray, brightness = self.preprocess(frame)
        preprocess_done = time.perf_counter()

        self.frame_index += 1
        rect = self._locate_face(gray, frame)
        locate_done = time.perf_counter()
        if rect is None:
            return self._no_face_metrics(
                brightness,
                total_started,
                preprocess_done,
                locate_done,
            )

        try:
            shape = self.predictor(gray, rect)
            raw_landmarks = shape_to_np(shape)
        except Exception:
            self._clear_tracking("landmark_error")
            return self._no_face_metrics(
                brightness,
                total_started,
                preprocess_done,
                locate_done,
            )

        if not self._landmarks_valid(raw_landmarks, gray.shape):
            self._clear_tracking("landmarks_invalidos")
            return self._no_face_metrics(
                brightness,
                total_started,
                preprocess_done,
                locate_done,
            )

        landmark_rect = self._rect_from_landmarks(raw_landmarks, gray.shape)
        rect = self._blend_rect(
            self.last_rect,
            landmark_rect,
            self.tracking_rect_alpha,
            gray.shape,
        )
        self.last_rect = rect
        landmarks = self._stabilize_landmarks(raw_landmarks)
        landmarks_done = time.perf_counter()

        raw_left_eye = raw_landmarks[LEFT_EYE]
        raw_right_eye = raw_landmarks[RIGHT_EYE]
        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]
        mouth = landmarks[MOUTH]
        left_ear = self.eye_aspect_ratio(raw_left_eye)
        right_ear = self.eye_aspect_ratio(raw_right_eye)
        ear_raw = (left_ear + right_ear) * 0.5
        mar = self.mouth_aspect_ratio(raw_landmarks[MOUTH])

        if self.last_pose[0] is None or self.frame_index % self.pose_interval == 0:
            self.last_pose = self.estimate_head_pose(landmarks, frame.shape)
        if self.last_gaze == "DESCONOCIDA" or self.frame_index % self.gaze_interval == 0:
            self.last_gaze = self.estimate_gaze(gray, landmarks)

        pitch, yaw, roll = self.last_pose
        quality = self.quality_score(rect, landmarks, gray.shape)
        eye_sharpness = self.eye_sharpness(gray, raw_left_eye, raw_right_eye)
        eye_ear_difference = abs(left_ear - right_ear)
        eye_width = min(
            point_distance(raw_left_eye[0], raw_left_eye[3]),
            point_distance(raw_right_eye[0], raw_right_eye[3]),
        )
        symmetry_limit = max(
            0.035,
            min(
                self.max_eye_ear_difference,
                self.max_eye_ear_difference_ratio * max(ear_raw, 1e-6),
            ),
        )
        eye_reliable = bool(
            quality >= 0.25
            and eye_width >= self.min_eye_width_pixels
            and eye_sharpness >= self.min_eye_sharpness
            and 0.04 <= ear_raw <= 0.60
            and eye_ear_difference <= symmetry_limit
            and (yaw is None or abs(float(yaw)) <= self.max_eye_yaw_degrees)
        )
        if eye_reliable:
            self.ear_history.append(ear_raw)
        if self.ear_history:
            ordered_ear = sorted(self.ear_history)
            ear = float(ordered_ear[len(ordered_ear) // 2])
        else:
            ear = ear_raw
        finished = time.perf_counter()
        timing = self._timing(
            total_started,
            preprocess_done,
            locate_done,
            landmarks_done,
            finished,
        )
        metrics = {
            "face_detected": True,
            "rect": (rect.left(), rect.top(), rect.right(), rect.bottom()),
            "landmarks": landmarks,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "mouth": mouth,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "ear_raw": ear_raw,
            "ear": ear,
            "eye_reliable": eye_reliable,
            "eye_sharpness": eye_sharpness,
            "eye_ear_difference": eye_ear_difference,
            "mar": mar,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "gaze": self.last_gaze,
            "quality": quality,
            "brightness": brightness,
            "analysis_source": self.last_detection_source,
            "analysis_breakdown_ms": timing,
        }
        metrics.update(self.face_detector.status())
        return metrics

    def _no_face_metrics(self, brightness, started, preprocess_done, locate_done):
        finished = time.perf_counter()
        timing = self._timing(
            started,
            preprocess_done,
            locate_done,
            locate_done,
            finished,
        )
        metrics = {
            "face_detected": False,
            "brightness": brightness,
            "quality": 0.0,
            "analysis_source": self.last_detection_source,
            "analysis_breakdown_ms": timing,
        }
        metrics.update(self.face_detector.status())
        return metrics

    @staticmethod
    def _timing(started, preprocess_done, locate_done, landmarks_done, finished):
        return {
            "preprocess": (preprocess_done - started) * 1000.0,
            "locate": (locate_done - preprocess_done) * 1000.0,
            "landmarks": (landmarks_done - locate_done) * 1000.0,
            "features": (finished - landmarks_done) * 1000.0,
            "total": (finished - started) * 1000.0,
        }

    def _locate_face(self, gray, frame=None):
        if self.last_rect is None:
            should_detect = self.frame_index == 1 or self.frame_index % self.no_face_detection_interval == 0
            if not should_detect:
                self.last_detection_source = "busqueda_espaciada"
                return None
            return self._detect_face(gray, frame)

        should_redetect = self.frame_index % self.detection_interval == 0
        if self.tracking_mode == "correlation" and self.tracker is not None and not should_redetect:
            quality = self.tracker.update(gray)
            if quality >= self.tracker_quality_threshold:
                position = self.tracker.get_position()
                self.last_rect = self._clamp_rect(
                    int(position.left()),
                    int(position.top()),
                    int(position.right()),
                    int(position.bottom()),
                    gray.shape,
                )
                self.last_detection_source = "correlation"
                return self.last_rect
            should_redetect = True

        if should_redetect:
            return self._detect_face(gray, frame)

        self.last_detection_source = "landmarks"
        return self.last_rect

    def _detect_face(self, gray, frame=None):
        previous = self.last_rect
        scale = self.detector_scale
        rects = self.face_detector.detect(gray, frame, scale)
        self.last_detection_source = self.face_detector.last_source

        if not rects:
            if previous is not None and self.redetection_miss_count < self.redetection_miss_tolerance:
                self.redetection_miss_count += 1
                self.last_rect = previous
                self.last_detection_source += "_retencion"
                return previous
            self._clear_tracking(self.last_detection_source)
            return None

        self.redetection_miss_count = 0
        if previous is None:
            rect = max(rects, key=rect_area)
        else:
            matched = max(rects, key=lambda candidate: self._rect_iou(previous, candidate))
            overlap = self._rect_iou(previous, matched)
            center_shift = self._normalized_center_shift(previous, matched)
            if overlap >= self.redetection_min_iou or center_shift <= self.redetection_max_center_shift:
                rect = self._blend_rect(
                    previous,
                    matched,
                    self.redetection_rect_alpha,
                    gray.shape,
                )
                self.last_detection_source += "_fusion"
            else:
                rect = max(rects, key=rect_area)
        self.last_rect = rect
        if self.tracking_mode == "correlation":
            self.tracker = dlib.correlation_tracker()
            self.tracker.start_track(gray, rect)
        else:
            self.tracker = None
        return rect

    def _clear_tracking(self, source):
        self.tracker = None
        self.last_rect = None
        self.filtered_landmarks = None
        self.ear_history.clear()
        self.redetection_miss_count = 0
        self.last_detection_source = source

    def reset_eye_filter(self):
        """Evita mezclar muestras EAR entre dos perfiles de calibracion."""
        self.ear_history.clear()

    def _stabilize_landmarks(self, points):
        current = np.asarray(points, dtype=np.float64)
        if self.filtered_landmarks is None or self.filtered_landmarks.shape != current.shape:
            self.filtered_landmarks = current.copy()
        else:
            previous = self.filtered_landmarks
            global_shift = np.median(current - previous, axis=0)
            translated = previous + self.landmark_translation_alpha * global_shift
            self.filtered_landmarks = translated + self.landmark_shape_alpha * (current - translated)
        return np.rint(self.filtered_landmarks).astype(np.int32)

    @staticmethod
    def _clamp01(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _rect_iou(first, second):
        left = max(first.left(), second.left())
        top = max(first.top(), second.top())
        right = min(first.right(), second.right())
        bottom = min(first.bottom(), second.bottom())
        intersection = max(0, right - left) * max(0, bottom - top)
        union = rect_area(first) + rect_area(second) - intersection
        return float(intersection) / float(union) if union > 0 else 0.0

    @staticmethod
    def _normalized_center_shift(first, second):
        first_x = (first.left() + first.right()) * 0.5
        first_y = (first.top() + first.bottom()) * 0.5
        second_x = (second.left() + second.right()) * 0.5
        second_y = (second.top() + second.bottom()) * 0.5
        scale = max(
            1.0,
            float(first.right() - first.left()),
            float(first.bottom() - first.top()),
        )
        return math.hypot(second_x - first_x, second_y - first_y) / scale

    @staticmethod
    def _blend_rect(previous, candidate, alpha, shape):
        if previous is None:
            return FaceAnalyzer._clamp_rect(
                candidate.left(), candidate.top(), candidate.right(), candidate.bottom(), shape
            )
        alpha = FaceAnalyzer._clamp01(alpha)
        beta = 1.0 - alpha
        return FaceAnalyzer._clamp_rect(
            beta * previous.left() + alpha * candidate.left(),
            beta * previous.top() + alpha * candidate.top(),
            beta * previous.right() + alpha * candidate.right(),
            beta * previous.bottom() + alpha * candidate.bottom(),
            shape,
        )

    @staticmethod
    def _clamp_rect(left, top, right, bottom, shape):
        height, width = shape[:2]
        left = max(0, min(width - 2, int(left)))
        top = max(0, min(height - 2, int(top)))
        right = max(left + 1, min(width - 1, int(right)))
        bottom = max(top + 1, min(height - 1, int(bottom)))
        return dlib.rectangle(left, top, right, bottom)

    @staticmethod
    def _rect_from_landmarks(points, shape):
        min_x = float(np.min(points[:, 0]))
        max_x = float(np.max(points[:, 0]))
        min_y = float(np.min(points[:, 1]))
        max_y = float(np.max(points[:, 1]))
        width = max(2.0, max_x - min_x)
        height = max(2.0, max_y - min_y)
        left = min_x - width * 0.15
        right = max_x + width * 0.15
        top = min_y - height * 0.38
        bottom = max_y + height * 0.12
        return FaceAnalyzer._clamp_rect(left, top, right, bottom, shape)

    @staticmethod
    def _landmarks_valid(points, shape):
        if points.shape != (68, 2) or not np.isfinite(points).all():
            return False
        height, width = shape[:2]
        eye_span = point_distance(points[36], points[45])
        face_width = float(np.max(points[:, 0]) - np.min(points[:, 0]))
        face_height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        center_x = float(np.median(points[:, 0]))
        center_y = float(np.median(points[:, 1]))
        return (
            eye_span >= 12.0
            and face_width >= 35.0
            and face_height >= 35.0
            and -width * 0.1 <= center_x <= width * 1.1
            and -height * 0.1 <= center_y <= height * 1.1
        )

    @staticmethod
    def eye_aspect_ratio(eye):
        a = point_distance(eye[1], eye[5])
        b = point_distance(eye[2], eye[4])
        c = point_distance(eye[0], eye[3])
        return float((a + b) / (2.0 * c)) if c > 1e-6 else 0.0

    @staticmethod
    def mouth_aspect_ratio(mouth):
        a = point_distance(mouth[13], mouth[19])
        b = point_distance(mouth[14], mouth[18])
        c = point_distance(mouth[15], mouth[17])
        d = point_distance(mouth[12], mouth[16])
        return float((a + b + c) / (3.0 * d)) if d > 1e-6 else 0.0

    @staticmethod
    def eye_sharpness(gray, left_eye, right_eye):
        points = np.vstack((left_eye, right_eye))
        x, y, width, height = cv2.boundingRect(points.astype(np.int32))
        pad_x = max(2, int(width * 0.12))
        pad_y = max(2, int(height * 0.50))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(gray.shape[1], x + width + pad_x)
        y1 = min(gray.shape[0], y + height + pad_y)
        roi = gray[y0:y1, x0:x1]
        if roi.size < 64:
            return 0.0
        return float(cv2.Laplacian(roi, cv2.CV_32F).var())

    def estimate_head_pose(self, points, frame_shape):
        image_points = np.array(
            [points[30], points[8], points[36], points[45], points[48], points[54]],
            dtype=np.float64,
        )
        height, width = frame_shape[:2]
        camera_shape = (height, width)
        if self.camera_shape != camera_shape:
            focal = float(width)
            self.camera_matrix = np.array([
                [focal, 0.0, width * 0.5],
                [0.0, focal, height * 0.5],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            self.camera_shape = camera_shape
        try:
            ok, rotation_vector, _ = cv2.solvePnP(
                self.MODEL_POINTS,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                return self.last_pose
            rotation, _ = cv2.Rodrigues(rotation_vector)
            projection = np.empty((3, 4), dtype=np.float64)
            projection[:, :3] = rotation
            projection[:, 3] = 0.0
            angles = cv2.decomposeProjectionMatrix(projection)[6]
            pitch, yaw, roll = [float(value) for value in angles.ravel()]
        except Exception:
            return self.last_pose
        self.angle_history.append((pitch, yaw, roll))
        values = np.asarray(self.angle_history, dtype=np.float64)
        return tuple(float(value) for value in np.median(values, axis=0))

    def estimate_gaze(self, gray, points):
        labels = []
        for indexes in (LEFT_EYE, RIGHT_EYE):
            polygon = points[indexes]
            x, y, width, height = cv2.boundingRect(polygon)
            if width < 8 or height < 4:
                continue
            pad = 2
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(gray.shape[1], x + width + pad)
            y1 = min(gray.shape[0], y + height + pad)
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            _, threshold = cv2.threshold(
                blur,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            moments = cv2.moments(threshold)
            if moments["m00"] <= 1.0:
                continue
            cx = (moments["m10"] / moments["m00"]) / max(1.0, float(x1 - x0))
            cy = (moments["m01"] / moments["m00"]) / max(1.0, float(y1 - y0))
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
        if not self.gaze_history:
            return label
        history = list(self.gaze_history)
        return max(set(history), key=history.count)

    @staticmethod
    def quality_score(rect, points, shape):
        height, width = shape[:2]
        inside = (
            rect.left() >= 0
            and rect.top() >= 0
            and rect.right() < width
            and rect.bottom() < height
        )
        area = float(rect_area(rect)) / float(max(1, width * height))
        eye_span = point_distance(points[36], points[45])
        score = min(1.0, area * 8.0) * (1.0 if inside else 0.6)
        if eye_span < 25.0:
            score *= 0.5
        return float(score)
