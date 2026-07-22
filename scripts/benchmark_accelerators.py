#!/usr/bin/env python3
import json
import os
import statistics
import time

import cv2
import dlib
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
FRAME = np.zeros((360, 640, 3), dtype=np.uint8)


def measure(callback, warmup=3, runs=30):
    for _ in range(warmup):
        callback()
    values = []
    for _ in range(runs):
        started = time.perf_counter()
        callback()
        values.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[int((len(ordered) - 1) * 0.95)], 3),
        "fps": round(1000.0 / statistics.mean(values), 3),
    }


def make_opencv_dnn(backend, target):
    net = cv2.dnn.readNetFromCaffe(
        os.path.join(MODELS, "deploy.prototxt"),
        os.path.join(MODELS, "res10_300x300_ssd_iter_140000_fp16.caffemodel"),
    )
    net.setPreferableBackend(backend)
    net.setPreferableTarget(target)

    def run():
        blob = cv2.dnn.blobFromImage(
            FRAME,
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )
        net.setInput(blob)
        return net.forward()

    return run


def main():
    results = {}
    results["opencv_dnn_cpu"] = measure(make_opencv_dnn(
        cv2.dnn.DNN_BACKEND_OPENCV,
        cv2.dnn.DNN_TARGET_CPU,
    ))
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        results["opencv_dnn_cuda_fp16"] = measure(make_opencv_dnn(
            cv2.dnn.DNN_BACKEND_CUDA,
            cv2.dnn.DNN_TARGET_CUDA_FP16,
        ))

    hog = dlib.get_frontal_face_detector()
    gray_small = cv2.resize(
        cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY),
        (320, 180),
        interpolation=cv2.INTER_AREA,
    )
    results["dlib_hog_half"] = measure(lambda: hog(gray_small, 0))

    cnn_path = os.path.join(MODELS, "mmod_human_face_detector.dat")
    if os.path.exists(cnn_path) and getattr(dlib, "DLIB_USE_CUDA", False):
        cnn = dlib.cnn_face_detection_model_v1(cnn_path)
        rgb_small = cv2.cvtColor(
            cv2.resize(FRAME, (320, 180), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        results["dlib_cnn_cuda_half"] = measure(
            lambda: cnn(rgb_small, 0),
            warmup=2,
            runs=15,
        )

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
