import os

import cv2
import dlib
import numpy as np


class FaceDetectorBackend(object):
    """Select the fastest available Jetson face detector with safe fallbacks."""

    SUPPORTED = (
        "dlib_cnn_cuda",
        "opencv_cuda_fp16",
        "dlib_hog",
    )

    def __init__(self, config):
        self.config = config
        detector_cfg = config.get("face_detection", {})
        dlib_cfg = config.get("dlib", {})
        self.requested = str(detector_cfg.get("backend", "auto")).lower()
        self.confidence_threshold = float(
            detector_cfg.get("confidence_threshold", 0.55)
        )
        self.allow_fallback = bool(detector_cfg.get("allow_fallback", True))
        configured_order = detector_cfg.get("backend_order", list(self.SUPPORTED))
        self.backend_order = [
            str(name).lower() for name in configured_order
            if str(name).lower() in self.SUPPORTED
        ]
        if not self.backend_order:
            self.backend_order = list(self.SUPPORTED)

        if self.requested == "auto":
            candidates = list(self.backend_order)
        elif self.requested in self.SUPPORTED:
            candidates = [self.requested]
            if self.allow_fallback:
                candidates.extend(
                    name for name in self.backend_order if name != self.requested
                )
        else:
            raise RuntimeError(
                "Backend facial no soportado: %s" % self.requested
            )

        self.candidates = candidates
        self.backend = None
        self.accelerated = False
        self.fallback_reason = ""
        self.last_source = "inicial"
        self._candidate_index = -1
        self._detector = None
        self._attempt_errors = []
        self._cnn_model_path = self._resolve_path(
            detector_cfg.get(
                "dlib_cnn_model_path",
                "models/mmod_human_face_detector.dat",
            )
        )
        self._opencv_config_path = self._resolve_path(
            detector_cfg.get(
                "opencv_config_path",
                "models/deploy.prototxt",
            )
        )
        self._opencv_model_path = self._resolve_path(
            detector_cfg.get(
                "opencv_model_path",
                "models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
            )
        )
        self._warmup = bool(detector_cfg.get("warmup", True))
        self._upsample = int(dlib_cfg.get("upsample", 0))
        self._activate_from(0)

    @staticmethod
    def _resolve_path(path):
        return path if os.path.isabs(path) else os.path.abspath(path)

    @staticmethod
    def _cuda_device_count():
        try:
            return int(cv2.cuda.getCudaEnabledDeviceCount())
        except Exception:
            return 0

    @staticmethod
    def _dlib_cuda_device_count():
        if not bool(getattr(dlib, "DLIB_USE_CUDA", False)):
            return 0
        try:
            return int(dlib.cuda.get_num_devices())
        except Exception:
            return 0

    def _activate_from(self, start_index):
        for index in range(start_index, len(self.candidates)):
            name = self.candidates[index]
            try:
                detector, accelerated = self._create(name)
                self._candidate_index = index
                self.backend = name
                self.accelerated = accelerated
                self._detector = detector
                if self._attempt_errors:
                    self.fallback_reason = "; ".join(self._attempt_errors)
                return
            except Exception as exc:
                self._attempt_errors.append("%s: %s" % (name, exc))
        raise RuntimeError(
            "No hay un detector facial utilizable (%s)"
            % "; ".join(self._attempt_errors)
        )

    def _create(self, name):
        if name == "dlib_cnn_cuda":
            if self._dlib_cuda_device_count() < 1:
                raise RuntimeError("dlib no tiene un dispositivo CUDA disponible")
            if not os.path.exists(self._cnn_model_path):
                raise RuntimeError("modelo CNN no encontrado")
            detector = dlib.cnn_face_detection_model_v1(self._cnn_model_path)
            if self._warmup:
                detector(np.zeros((180, 320, 3), dtype=np.uint8), 0)
            return detector, True

        if name == "opencv_cuda_fp16":
            if self._cuda_device_count() < 1:
                raise RuntimeError("OpenCV no tiene un dispositivo CUDA disponible")
            if not os.path.exists(self._opencv_config_path):
                raise RuntimeError("deploy.prototxt no encontrado")
            if not os.path.exists(self._opencv_model_path):
                raise RuntimeError("modelo Caffe FP16 no encontrado")
            detector = cv2.dnn.readNetFromCaffe(
                self._opencv_config_path,
                self._opencv_model_path,
            )
            detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            if self._warmup:
                blob = cv2.dnn.blobFromImage(
                    np.zeros((180, 320, 3), dtype=np.uint8),
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0),
                    swapRB=False,
                    crop=False,
                )
                detector.setInput(blob)
                detector.forward()
            return detector, True

        if name == "dlib_hog":
            return dlib.get_frontal_face_detector(), False

        raise RuntimeError("backend desconocido")

    def detect(self, gray, frame=None, scale=0.5):
        try:
            return self._detect_current(gray, frame, scale)
        except Exception as exc:
            failed_backend = self.backend
            self._attempt_errors.append(
                "%s fallo en ejecucion: %s" % (failed_backend, exc)
            )
            if not self.allow_fallback:
                raise
            self._activate_from(self._candidate_index + 1)
            return self._detect_current(gray, frame, scale)

    def _detect_current(self, gray, frame, scale):
        if self.backend == "opencv_cuda_fp16":
            rects = self._detect_opencv(frame, gray)
            self.last_source = self.backend
            return rects

        if self.backend == "dlib_cnn_cuda":
            source = frame if frame is not None else gray
            small_source = self._resize(source, scale)
            rgb = self._to_rgb(small_source)
            detections = self._detector(rgb, self._upsample)
            rects = [item.rect for item in detections]
        else:
            small_gray = self._resize(gray, scale)
            rects = list(self._detector(small_gray, self._upsample))

        self.last_source = "%s_%.2f" % (self.backend, scale)
        return self._restore_scale(rects, gray.shape, scale)

    @staticmethod
    def _resize(image, scale):
        if scale >= 0.999:
            return image
        return cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _to_rgb(frame):
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _detect_opencv(self, frame, gray):
        if frame is None:
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr = frame
        blob = cv2.dnn.blobFromImage(
            bgr,
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )
        self._detector.setInput(blob)
        output = self._detector.forward().reshape(-1, 7)
        height, width = gray.shape[:2]
        rects = []
        for detection in output:
            confidence = float(detection[2])
            if confidence < self.confidence_threshold:
                continue
            rects.append(self._clamp_rect(
                detection[3] * width,
                detection[4] * height,
                detection[5] * width,
                detection[6] * height,
                gray.shape,
            ))
        return rects

    @classmethod
    def _restore_scale(cls, rects, shape, scale):
        if scale >= 0.999:
            return [
                cls._clamp_rect(
                    rect.left(), rect.top(), rect.right(), rect.bottom(), shape
                )
                for rect in rects
            ]
        inverse = 1.0 / scale
        return [
            cls._clamp_rect(
                round(rect.left() * inverse),
                round(rect.top() * inverse),
                round(rect.right() * inverse),
                round(rect.bottom() * inverse),
                shape,
            )
            for rect in rects
        ]

    @staticmethod
    def _clamp_rect(left, top, right, bottom, shape):
        height, width = shape[:2]
        left = max(0, min(width - 2, int(left)))
        top = max(0, min(height - 2, int(top)))
        right = max(left + 1, min(width - 1, int(right)))
        bottom = max(top + 1, min(height - 1, int(bottom)))
        return dlib.rectangle(left, top, right, bottom)

    def status(self):
        return {
            "face_detector_requested": self.requested,
            "face_detector_backend": self.backend,
            "face_detector_accelerated": self.accelerated,
            "face_detector_fallback": self.fallback_reason,
        }
