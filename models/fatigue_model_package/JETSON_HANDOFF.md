# JETSON AGENT HANDOFF DOCUMENTATION

**Target Device**: NVIDIA Jetson Nano Developer Kit (4 GB RAM)
**Package Name**: `fatigue_model_package`
**Exporter Environment**: Windows 11 host (CPU mode)

---

## 1. Files to Copy to NVIDIA Jetson

Copy the contents of the `export/` directory to the Jetson project repository:

* `export/eye_state.onnx`
* `export/yawn_state.onnx`
* `export/head_pose.onnx`
* `export/model_manifest.json`
* `export/recommended_thresholds.json`
* `export/preprocessing_reference.py`

---

## 2. Model Specifications & SHA-256 Hashes

Refer to `export/model_manifest.json` for live dynamically-generated hashes.

### Model A: Eye State (`eye_state.onnx`)
* **Input Tensor**: `input` -> Shape: `[1, 1, 64, 64]` -> DataType: `Float32`
* **Preprocessing**: Grayscale, resized to 64x64, normalized to `[0.0, 1.0]`.
* **Output Tensor**: `output` -> Logits of shape `[1, 2]`
* **Class Ordering**: Index `0` -> `closed`, Index `1` -> `open`
* **Logits to Probabilities**:
  $$\text{P(closed)} = \frac{e^{z_0}}{e^{z_0} + e^{z_1}}$$

### Model B: Yawn State (`yawn_state.onnx`)
* **Input Tensor**: `input` -> Shape: `[1, 1, 64, 96]` -> DataType: `Float32`
* **Preprocessing**: Grayscale, resized to 64x96 (H=64, W=96), normalized to `[0.0, 1.0]`.
* **Output Tensor**: `output` -> Logits of shape `[1, 2]`
* **Class Ordering**: Index `0` -> `normal`, Index `1` -> `yawning`
* **Logits to Probabilities**:
  $$\text{P(yawning)} = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$$

### Model C: Head Pose Orientation (`head_pose.onnx`)
* **Input Tensor**: `input` -> Shape: `[1, 3, 128, 128]` -> DataType: `Float32`
* **Preprocessing**: RGB, resized to 128x128, normalized to `[0.0, 1.0]`.
* **Output Tensor**: `output` -> Continuous angles array of shape `[1, 3]`
* **Output Order**: `[pitch_deg, yaw_deg, roll_deg]` in degrees.

---

## 3. Strict Rules for Jetson Agent

1. **TensorRT Engine Generation**:
   - **DO NOT** use `.engine` files compiled on Windows or x86 host.
   - **MUST** build TensorRT `.engine` models directly on the NVIDIA Jetson Nano using `trtexec` or TensorRT C++/Python API.
   - Command reference to run on Jetson Nano:
     ```bash
     trtexec --onnx=eye_state.onnx --saveEngine=eye_state_fp16.engine --fp16
     trtexec --onnx=yawn_state.onnx --saveEngine=yawn_state_fp16.engine --fp16
     trtexec --onnx=head_pose.onnx --saveEngine=head_pose_fp16.engine --fp16
     ```
2. **Numerical Output Validation**:
   - Compare outputs of `.onnx` (ONNX Runtime) vs `.engine` (TensorRT FP16) on Jetson using identical sample inputs.
   - Ensure maximum absolute difference is < 1e-2 before live camera integration.
3. **Hardware Benchmarking**:
   - Measure and document physical hardware performance on Jetson Nano:
     * Inference Latency (ms)
     * Frame Rate (FPS)
     * VRAM / System Memory Consumption (MB)
     * SoC Temperature (°C) under sustained execution.
4. **Temporal Logic Integration**:
   - Use `recommended_thresholds.json` as baseline classification probabilities.
   - Maintain state history over time (e.g. eye closed for > 1.5s, yawning for > 2.0s, head pitched down > 25° for > 1.0s) to trigger buzzer/LED alert states in main application.
