import os

import cv2
import numpy as np


class FatigueModelPackage(object):
    """Inferencia ONNX opcional y tolerante a fallos."""

    MODEL_FILES = {"eye_state": "eye_state.onnx", "yawn_state": "yawn_state.onnx", "head_pose": "head_pose.onnx"}

    def __init__(self, config):
        cfg = config.get("fatigue_models", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.directory = os.path.abspath(cfg.get("package_path", "models/fatigue_model_package"))
        self.interval = max(1, int(cfg.get("inference_interval_frames", 2)))
        self.eye_threshold = float(cfg.get("eye_closed_threshold", 0.5))
        self.yawn_threshold = float(cfg.get("yawn_threshold", 0.5))
        self.frame_index, self.nets, self.errors, self.last_predictions = 0, {}, {}, {}
        if self.enabled:
            self._load()

    def _load(self):
        for name, filename in self.MODEL_FILES.items():
            try:
                self.nets[name] = cv2.dnn.readNetFromONNX(os.path.join(self.directory, filename))
            except Exception as exc:
                self.errors[name] = str(exc)

    @staticmethod
    def _softmax(logits):
        values = np.asarray(logits, dtype=np.float32).reshape(-1)
        values -= np.max(values)
        values = np.exp(values)
        return values / max(float(np.sum(values)), 1e-8)

    @staticmethod
    def _crop(image, points, pad_x, pad_y):
        x, y, width, height = cv2.boundingRect(np.asarray(points, dtype=np.int32))
        px, py = max(2, int(width * pad_x)), max(2, int(height * pad_y))
        crop = image[max(0, y-py):min(image.shape[0], y+height+py), max(0, x-px):min(image.shape[1], x+width+px)]
        return crop if crop.size else None

    @staticmethod
    def _tensor(image, width, height, rgb=False):
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if rgb:
            if resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            elif resized.shape[2] == 4:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
            else:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            return resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        if resized.ndim == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY if resized.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        return resized[None, None].astype(np.float32) / 255.0

    def _forward(self, name, tensor):
        net = self.nets.get(name)
        if net is None:
            return None
        try:
            net.setInput(tensor)
            return net.forward()
        except Exception as exc:
            self.errors[name] = str(exc)
            self.nets.pop(name, None)
            return None

    def status(self):
        return {"fatigue_models_enabled": self.enabled, "fatigue_models_available": sorted(self.nets), "fatigue_models_errors": dict(self.errors)}

    def analyze(self, frame, landmarks, rect):
        self.frame_index += 1
        result = self.status()
        if not self.enabled or not self.nets:
            return result
        if self.frame_index % self.interval and self.last_predictions:
            result.update(self.last_predictions)
            return result
        predictions, eye_probabilities = {}, []
        for points in (landmarks[42:48], landmarks[36:42]):
            crop = self._crop(frame, points, 0.20, 0.70)
            output = self._forward("eye_state", self._tensor(crop, 64, 64)) if crop is not None else None
            if output is not None:
                eye_probabilities.append(float(self._softmax(output)[0]))
        if eye_probabilities:
            probability = float(sum(eye_probabilities) / len(eye_probabilities))
            predictions.update(model_eye_closed_probability=probability, model_eyes_closed=probability >= self.eye_threshold)
        mouth = self._crop(frame, landmarks[48:68], 0.18, 0.35)
        output = self._forward("yawn_state", self._tensor(mouth, 96, 64)) if mouth is not None else None
        if output is not None:
            probability = float(self._softmax(output)[1])
            predictions.update(model_yawn_probability=probability, model_yawning=probability >= self.yawn_threshold)
        left, top, right, bottom = rect
        face = frame[max(0, top):min(frame.shape[0], bottom), max(0, left):min(frame.shape[1], right)]
        output = self._forward("head_pose", self._tensor(face, 128, 128, rgb=True)) if face.size else None
        if output is not None:
            angles = np.asarray(output, dtype=np.float32).reshape(-1)
            if angles.size >= 3:
                predictions.update(model_pitch=float(angles[0]), model_yaw=float(angles[1]), model_roll=float(angles[2]))
        self.last_predictions = predictions
        result = self.status()
        result.update(predictions)
        return result
