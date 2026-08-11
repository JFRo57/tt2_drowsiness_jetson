# TT2 Drowsiness Jetson — versión Gamma

Rama: `gamma`
Base: `v4`, commit `572a7db`
Estado: experimental, con pruebas automatizadas aprobadas y validación física
pendiente en NVIDIA Jetson Nano.

## Alcance

Gamma conserva la arquitectura de V4 y añade personalización de bostezos,
continuidad temporal de cierres oculares y modelos ONNX complementarios. EAR,
MAR, PERCLOS, landmarks y PnP continúan siendo la base de decisión. Gamma no
es una certificación médica ni automotriz.

## Cambios por componente

### Calibración personal de bostezos

- Nueva etapa `YAWN` después de ojos y parpadeos.
- Exige ciclos completos, duración mínima y apertura diferenciable del MAR
  basal.
- Calcula umbrales personales de boca cerrada, abierta y ampliamente abierta.
- Persiste basal, pico, dispersión, eventos y umbrales.
- Permite recalibrar solo bostezos con la tecla `Y`.
- Una etapa inválida no reemplaza un perfil válido.

Archivos: `src/calibration.py`, `src/application.py`,
`src/presentation_ui.py`, `src/state_machine_ui.py`, `config.json` y
`tests/test_calibration.py`.

### Continuidad de cierres oculares

- Se separaron umbrales de entrada, mantenimiento y liberación.
- Una caída breve de medición conserva el episodio de cierre profundo.
- Una medición inválida breve pausa la evidencia; no se interpreta como ojos
  abiertos.
- Al agotar la tolerancia, el episodio termina de forma determinista.
- CSV y telemetría exponen estado y duración del dropout.

Archivos: `src/temporal_events.py`, `src/fatigue_detector.py`,
`src/event_logger.py`, `config.json` y
`tests/test_temporal_state_machine.py`.

### Modelos ONNX Gamma

| Modelo | Entrada | Salida |
| --- | --- | --- |
| `eye_state.onnx` | gris `1x1x64x64`, por ojo | probabilidad de cierre |
| `yawn_state.onnx` | gris `1x1x64x96` | probabilidad de bostezo |
| `head_pose.onnx` | RGB `1x3x128x128` | pitch, yaw y roll |

`src/fatigue_models.py` usa OpenCV DNN y respeta el manifiesto del paquete.
Ejecuta inferencia cada dos análisis por defecto y reutiliza el resultado en el
intervalo. Cada red se carga y desactiva independientemente: un archivo ausente
o incompatible no detiene el monitoreo geométrico. Las salidas son evidencia
complementaria y telemetría, no decisiones autónomas.

Métricas publicadas:

- `model_eye_closed_probability` y `model_eyes_closed`;
- `model_yawn_probability` y `model_yawning`;
- `model_pitch`, `model_yaw` y `model_roll`;
- `fatigue_models_available` y `fatigue_models_errors`.

Archivos: `src/fatigue_models.py`, `src/face_analyzer.py`,
`models/fatigue_model_package/`, `config.json` y
`tests/test_fatigue_models.py`.

### Redetección facial

Se conserva la búsqueda acelerada sin rostro, la tolerancia limitada a fallos,
el descarte del ROI obsoleto y la fusión controlada al reaparecer el rostro.
Las etapas de calibración con ojos cerrados mantienen el bloqueo previsto.

### Hardware y observabilidad

- LEDs BOARD: verde 11, amarillo 13 y rojo 15.
- Switch BOARD: automático 12 y emergencia 16.
- CSV: dropout ocular, estado/duración del bostezo y MAR máximo.
- UI: progreso y resultados de calibración de bostezos.

Verificar estos pines contra el cableado antes del primer arranque.

## Configuración ONNX

```json
"fatigue_models": {
  "enabled": true,
  "package_path": "models/fatigue_model_package",
  "inference_interval_frames": 2,
  "eye_closed_threshold": 0.5,
  "yawn_threshold": 0.5
}
```

## Validación

```bash
python3 -m json.tool config.json
python3 -m py_compile main.py src/*.py
python3 -m unittest discover -s tests
git diff --check
```

Antes de publicar: 81 pruebas aprobadas y los tres ONNX cargados con una
inferencia de humo real mediante OpenCV DNN.

## Instalación de Gamma

```bash
git clone --branch gamma https://github.com/JFRo57/tt2_drowsiness_jetson.git \
  tt2_drowsiness_jetson_gamma
cd tt2_drowsiness_jetson_gamma
python3 -m unittest discover -s tests
python3 main.py --gpio-self-test
python3 main.py
```

No se distribuyen motores TensorRT. Deben construirse y validarse en la Jetson
destino antes de sustituir los ONNX.

## Validación física pendiente

1. Confirmar cableado BOARD.
2. Medir FPS, memoria y temperatura con rostro y ONNX activos.
3. Probar pérdida y recuperación facial con distintas iluminaciones.
4. Calibrar ojos, parpadeos y bostezos con varios usuarios.
5. Comparar ONNX contra datos etiquetados antes de ajustar umbrales.
6. Probar solo en simulador o vehículo inmóvil.
