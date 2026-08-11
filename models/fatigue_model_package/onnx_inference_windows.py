import os
import json
import cv2
import numpy as np
import onnxruntime as ort
from preprocessing_reference import preprocess_eye, preprocess_yawn, preprocess_head_pose

def softmax(logits: np.ndarray) -> np.ndarray:
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class WindowsModelTester:
    def __init__(self, export_dir: str = "."):
        manifest_path = os.path.join(export_dir, "model_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.export_dir = export_dir

        eye_model_path = os.path.join(export_dir, self.manifest["models"]["eye_state"]["model_file"])
        yawn_model_path = os.path.join(export_dir, self.manifest["models"]["yawn_state"]["model_file"])
        head_model_path = os.path.join(export_dir, self.manifest["models"]["head_pose"]["model_file"])

        self.eye_sess = ort.InferenceSession(eye_model_path, providers=["CPUExecutionProvider"])
        self.yawn_sess = ort.InferenceSession(yawn_model_path, providers=["CPUExecutionProvider"])
        self.head_sess = ort.InferenceSession(head_model_path, providers=["CPUExecutionProvider"])

        self.eye_classes = self.manifest["models"]["eye_state"]["classes"]
        self.yawn_classes = self.manifest["models"]["yawn_state"]["classes"]

    def predict_eye(self, image: np.ndarray):
        input_tensor = preprocess_eye(image)
        input_name = self.eye_sess.get_inputs()[0].name
        logits = self.eye_sess.run(None, {input_name: input_tensor})[0]
        probs = softmax(logits)[0]
        pred_idx = int(np.argmax(probs))
        return {
            "predicted_class": self.eye_classes[pred_idx],
            "class_probabilities": {c: float(probs[i]) for i, c in enumerate(self.eye_classes)},
            "raw_logits": logits[0].tolist()
        }

    def predict_yawn(self, image: np.ndarray):
        input_tensor = preprocess_yawn(image)
        input_name = self.yawn_sess.get_inputs()[0].name
        logits = self.yawn_sess.run(None, {input_name: input_tensor})[0]
        probs = softmax(logits)[0]
        pred_idx = int(np.argmax(probs))
        return {
            "predicted_class": self.yawn_classes[pred_idx],
            "class_probabilities": {c: float(probs[i]) for i, c in enumerate(self.yawn_classes)},
            "raw_logits": logits[0].tolist()
        }

    def predict_head_pose(self, image: np.ndarray):
        input_tensor = preprocess_head_pose(image)
        input_name = self.head_sess.get_inputs()[0].name
        angles = self.head_sess.run(None, {input_name: input_tensor})[0][0]
        return {
            "pitch_deg": float(angles[0]),
            "yaw_deg": float(angles[1]),
            "roll_deg": float(angles[2])
        }

def main():
    export_dir = os.path.dirname(os.path.abspath(__file__))
    tester = WindowsModelTester(export_dir=export_dir)

    print("=== Testing Windows ONNX Runtime Inference ===")

    # Test Eye Model with synthetic sample image
    dummy_eye_img = np.full((64, 64), 128, dtype=np.uint8)
    eye_res = tester.predict_eye(dummy_eye_img)
    print("Eye State Prediction:", eye_res)

    # Test Yawn Model with synthetic sample image
    dummy_yawn_img = np.full((64, 96), 128, dtype=np.uint8)
    yawn_res = tester.predict_yawn(dummy_yawn_img)
    print("Yawn State Prediction:", yawn_res)

    # Test Head Pose Model with synthetic sample image
    dummy_head_img = np.full((128, 128, 3), 180, dtype=np.uint8)
    head_res = tester.predict_head_pose(dummy_head_img)
    print("Head Pose Prediction:", head_res)

if __name__ == "__main__":
    main()
