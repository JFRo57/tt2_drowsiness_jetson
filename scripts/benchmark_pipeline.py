#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import cv2


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import load_config
from src.camera import CameraManager
from src.face_analyzer import FaceAnalyzer


def percentile(values, percent):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * (float(percent) / 100.0)))
    return float(ordered[max(0, min(len(ordered) - 1, index))])


def main():
    parser = argparse.ArgumentParser(description="Benchmark del pipeline de vision sin interfaz ni GPIO")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    args = parser.parse_args()

    os.chdir(ROOT)
    config = load_config(args.config)
    camera = CameraManager(config["camera"])
    analyzer = FaceAnalyzer(config)
    latencies = []
    faces = 0
    unique_frames = 0
    last_sequence = -1
    source_counts = {}
    stage_totals = {}
    started = 0.0

    if not camera.start():
        print(json.dumps({"error": camera.error or "No se pudo abrir la camara"}))
        return 1
    detector_status = analyzer.face_detector.status()
    try:
        warmup_deadline = time.monotonic() + max(0.0, args.warmup)
        while time.monotonic() < warmup_deadline:
            frame, _, sequence = camera.wait_for_frame(
                last_sequence,
                timeout=0.2,
                copy=False,
            )
            if frame is None:
                continue
            last_sequence = sequence
            analyzer.analyze(frame)

        started = time.monotonic()
        deadline = started + max(1.0, args.seconds)
        while time.monotonic() < deadline:
            frame, _, sequence = camera.wait_for_frame(
                last_sequence,
                timeout=0.2,
                copy=False,
            )
            if frame is None:
                continue
            last_sequence = sequence
            analysis_started = time.monotonic()
            metrics = analyzer.analyze(frame)
            latencies.append((time.monotonic() - analysis_started) * 1000.0)
            unique_frames += 1
            faces += int(bool(metrics.get("face_detected")))
            source = metrics.get("analysis_source", "desconocida")
            source_counts[source] = source_counts.get(source, 0) + 1
            for name, value in metrics.get("analysis_breakdown_ms", {}).items():
                stage_totals[name] = stage_totals.get(name, 0.0) + float(value)
    finally:
        camera.stop()

    elapsed = max(1e-6, time.monotonic() - started)
    result = {
        "elapsed_seconds": round(elapsed, 3),
        "frames_analyzed": unique_frames,
        "analysis_fps": round(unique_frames / elapsed, 3),
        "capture_fps": round(camera.capture_fps, 3),
        "face_detection_rate": round(float(faces) / max(1, unique_frames), 3),
        "latency_ms_mean": round(sum(latencies) / max(1, len(latencies)), 3),
        "latency_ms_p50": round(percentile(latencies, 50), 3),
        "latency_ms_p95": round(percentile(latencies, 95), 3),
        "latency_ms_max": round(max(latencies) if latencies else 0.0, 3),
        "analysis_sources": source_counts,
        "stage_ms_mean": {
            name: round(total / max(1, unique_frames), 3)
            for name, total in sorted(stage_totals.items())
        },
        "camera_output_format": config["camera"].get("output_format", "BGRx"),
        "face_detector": detector_status,
        "opencv_cuda_devices": int(cv2.cuda.getCudaEnabledDeviceCount()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
