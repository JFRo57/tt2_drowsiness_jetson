# TT2 Drowsiness Jetson V2

> **Detección de somnolencia optimizada para NVIDIA Jetson Nano, con calibración personal, interfaz en tiempo real y alertas GPIO.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NVIDIA Jetson Nano](https://img.shields.io/badge/NVIDIA-Jetson%20Nano-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded/jetson-nano)
[![OpenCV](https://img.shields.io/badge/OpenCV-GStreamer-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![dlib](https://img.shields.io/badge/dlib-68%20landmarks-0080FF)](http://dlib.net/)
[![GPIO](https://img.shields.io/badge/GPIO-PWM2-orange)](https://github.com/NVIDIA/jetson-gpio)

---

## 📌 Descripción general

**TT2 Drowsiness Jetson V2** es un sistema embebido para detectar señales de
somnolencia y distracción mediante una cámara CSI. Procesa video en tiempo real,
localiza el rostro y sus 68 puntos faciales, calcula métricas como **EAR**,
**MAR**, **PERCLOS**, pose de cabeza y dirección de mirada, y clasifica el nivel
de riesgo del usuario.

Está dirigido a proyectos académicos, prototipos de asistencia al conductor y
desarrolladores que trabajen con visión artificial en **NVIDIA Jetson Nano**.
Además de la interfaz visual, puede activar **LEDs**, un **buzzer pasivo PWM** y
un switch físico con modos automático, mantenimiento y paro de emergencia.

### Características principales

- Captura de cámara CSI mediante **GStreamer** y `nvarguscamerasrc`.
- Detección facial y estimación de **68 landmarks** con dlib.
- Cálculo de **EAR**, **MAR**, **PERCLOS**, mirada y pose de cabeza.
- Detección temporal de ojos cerrados, bostezos, cabeceo y pérdida de rostro.
- Estados escalonados: normal, prealerta, alerta y alerta crítica.
- Calibración supervisada de tres perfiles personales: **ojos abiertos**,
  **posible somnolencia** y **dormido**, durante la sesión.
- Interfaz OpenCV con métricas, landmarks, FPS y estado del hardware.
- Modos de operación **AUTOMATIC**, **MAINTENANCE** y **EMERGENCY**.
- Ejecución con GPIO físico o en modo completamente simulado.
- Alertas con LEDs y buzzer pasivo **MH-FMD/YL-44 low-level trigger**.
- PWM2 nativo para mantener el tono estable bajo carga de OpenCV y dlib.
- Pipeline **latest-only**: nunca procesa dos veces el mismo cuadro ni acumula
  video atrasado.
- Detección facial a media escala, seguimiento ligero por landmarks y
  redetección periódica.
- CLAHE adaptativo, pose/mirada desacopladas y PERCLOS incremental.
- Objetivo de análisis configurable de **20 FPS** y telemetría por etapa en la
  interfaz.
- Registro opcional de transiciones de estado en CSV.
- Pruebas unitarias de calibración, estados, GPIO/PWM y ruta optimizada.

---

## 🛠️ Stack tecnológico

| Categoría | Tecnología |
| :--- | :--- |
| **Backend / Core** | Python 3 |
| **Visión artificial** | OpenCV, dlib, NumPy |
| **Captura de video** | Cámara CSI, GStreamer, `nvarguscamerasrc` |
| **Hardware** | NVIDIA Jetson Nano 4 GB, Jetson.GPIO, GPIO BOARD, PWM2 |
| **Interfaz** | Ventanas y overlays de OpenCV |
| **Servicios del sistema** | systemd, shell POSIX, sysfs PWM |
| **Pruebas** | `unittest`, `py_compile`, validación JSON |

---

## 🧩 Arquitectura

```mermaid
flowchart LR
    CAM[Cámara CSI] --> GST[GStreamer]
    GST --> APP[DrowsinessApplication]
    APP --> FACE[OpenCV + dlib]
    FACE --> FATIGUE[FatigueDetector]
    FATIGUE --> ALERT[AlertController]
    SWITCH[Switch de 3 estados] --> GPIO[GPIOController]
    ALERT --> GPIO
    GPIO --> LED[LEDs]
    GPIO --> BUZZER[Buzzer PWM2]
    APP --> UI[PresentationUI]
    APP --> LOG[EventLogger CSV]
```

La descripción detallada de módulos, estados y flujo de ejecución se encuentra
en [DOCUMENTACION_PROYECTO.md](DOCUMENTACION_PROYECTO.md).

---

## ⚡ Rendimiento de la V2

La optimización conserva el predictor dlib de 68 puntos y todo el hardware de
la versión base. El trabajo se reduce en los lugares de mayor costo:

- Captura en hilo independiente con espera por secuencia y acceso sin copia.
- Detección HOG a escala `0.5`, cada 3 análisis durante búsqueda y cada 12 con
  un rostro seguido.
- Seguimiento del ROI mediante landmarks entre redetecciones.
- Pose y mirada cada 2 análisis, reutilizando el último valor válido.
- CLAHE únicamente cuando la luminosidad sale del rango configurado.
- Escrituras LED agrupadas, panel gráfico reutilizable y PERCLOS incremental.

Benchmark realizado en esta Jetson Nano, cámara CSI a procesamiento
**640×360**, escena sin rostro:

| Métrica | Antes | V2 optimizada | Cambio |
| :--- | ---: | ---: | ---: |
| FPS del analizador | 7.812 | 29.328 | **3.75×** |
| Latencia media | 122.899 ms | 11.267 ms | **−90.8 %** |
| Latencia p95 | 125.718 ms | 33.328 ms | **−73.5 %** |

La aplicación limita deliberadamente el análisis a **20 FPS** para reservar
CPU a la interfaz y a GPIO. Los resultados dependen de iluminación, presencia
del rostro, temperatura y modo de energía. Para repetir la medición:

```bash
python3 scripts/benchmark_pipeline.py --seconds 10 --warmup 2
```

---

## 🚀 Requisitos previos e instalación

### Requisitos de hardware

- **NVIDIA Jetson Nano 4 GB**.
- Cámara CSI compatible con `nvarguscamerasrc`, por ejemplo **IMX219-77IR**.
- Opcional: buzzer pasivo de tres pines **MH-FMD/YL-44 o YL-44**.
- Opcional: tres LEDs con resistencias de **220–330 Ω**.
- Opcional: switch físico **ON-OFF-ON** de tres posiciones.

### Requisitos de software

- **JetPack / L4T** instalado y funcional.
- **Python 3**.
- OpenCV compilado con soporte para **GStreamer**.
- **dlib** y **NumPy**.
- **Jetson.GPIO** para utilizar el hardware físico.
- `wget` y `bzip2` para descargar el predictor facial.

> [!IMPORTANT]
> `requirements.txt` está vacío actualmente. En Jetson se recomienda conservar
> las versiones de OpenCV, NumPy y GStreamer proporcionadas por JetPack, en vez
> de sustituirlas indiscriminadamente con paquetes de `pip`.

### Pasos de instalación

1. **Clonar el repositorio:**

   ```bash
   git clone --branch v2 https://github.com/JFRo57/tt2_drowsiness_jetson.git tt2_drowsiness_jetson_v2
   cd tt2_drowsiness_jetson_v2
   ```

2. **Crear un entorno virtual reutilizando los paquetes de JetPack:**

   ```bash
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   ```

3. **Verificar las dependencias principales:**

   ```bash
   python3 -c "import cv2, dlib, numpy; print('OpenCV:', cv2.__version__, '| dlib:', dlib.__version__)"
   python3 -c "import Jetson.GPIO as GPIO; print('Jetson.GPIO:', GPIO.VERSION)"
   ```

4. **Descargar el predictor de 68 landmarks:**

   ```bash
   mkdir -p models
   wget https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -P models
   bzip2 -d models/shape_predictor_68_face_landmarks.dat.bz2
   ```

   El archivo `models/shape_predictor_68_face_landmarks.dat` ocupa
   aproximadamente **96 MB** y se excluye del repositorio mediante `.gitignore`.

5. **Comprobar el modelo y ejecutar una prueba en simulación:**

   ```bash
   test -f models/shape_predictor_68_face_landmarks.dat
   python3 main.py --presentation --simulation
   ```

---

## 🔌 Conexiones GPIO

El proyecto usa numeración física **BOARD**, no BCM.

| Función | Pin BOARD | Conexión |
| :--- | :---: | :--- |
| LED verde | 29 | Ánodo mediante resistencia de 220–330 Ω |
| LED amarillo | 31 | Ánodo mediante resistencia de 220–330 Ω |
| LED rojo | 32 | Ánodo mediante resistencia de 220–330 Ω |
| Buzzer pasivo | 33 | Pin `SIG` / `S` del módulo; función PWM2 |
| Switch automático | 35 | Extremo AUTO; activo hacia GND |
| Switch emergencia | 37 | Extremo EMERGENCY; activo hacia GND |
| Tierra común | GND | LEDs, buzzer y switch |

> [!CAUTION]
> Verifica voltajes, consumo, resistencias y pinout antes de energizar el
> circuito. No conectes una carga que exceda la corriente permitida por el GPIO.

---

## 🔊 Configuración de PWM2 y buzzer

El módulo probado es un buzzer pasivo **low-level trigger**: una salida **LOW**
lo activa y una salida **HIGH** lo silencia. Necesita una onda cuadrada de
aproximadamente **2–5 kHz**.

### 1. Habilitar PWM2 en Jetson-IO

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

Selecciona **`pwm2 (33)`**, guarda la configuración y reinicia la Jetson.

### 2. Instalar el inicializador incluido

En **L4T 32.7.6**, Jetson-IO puede mostrar PWM2 correctamente sin que BOARD 33
entregue todavía la señal física. El proyecto incluye un servicio que aplica el
pinmux necesario al arrancar y deja el buzzer en reposo HIGH:

```bash
sudo install -m 755 scripts/configure_pwm2_jetson_nano.sh /usr/local/sbin/tt2-configure-pwm2
sudo install -m 644 systemd/tt2-pwm2-pinmux.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tt2-pwm2-pinmux.service
```

### 3. Verificar PWM y GPIO

```bash
systemctl status tt2-pwm2-pinmux.service --no-pager
python3 main.py --gpio-self-test
```

La salida debe indicar **`Backend buzzer: PWM nativo`**, reproducir dos tonos y
quedar en silencio al finalizar.

### Por qué no se utiliza `PWM.stop()`

El backend controla PWM2 mediante `/sys/class/pwm`:

- **Tono:** ciclo de trabajo del 50 %.
- **Silencio:** ciclo de trabajo del 100 %, equivalente a salida HIGH.
- **Cleanup:** el canal queda habilitado en HIGH.

`Jetson.GPIO.PWM.stop()` puede dejar la salida en LOW y mantener este módulo
sonando. Por ello no debe utilizarse sobre BOARD 33 con este buzzer.

---

## ▶️ Ejecución

Activa siempre el entorno virtual independiente de esta versión:

```bash
cd /home/rafael/Documentos/tt2_drowsiness_jetson_v2
source .venv/bin/activate
```

### Interfaz con hardware simulado

```bash
python3 main.py --presentation --simulation
```

### Ejecución headless simulada

```bash
python3 main.py --headless --simulation
```

### Ejecución headless con GPIO físico

```bash
python3 main.py --headless
```

### Interfaz con GPIO físico

Configura `gpio.enabled: true` y `gpio.simulation_mode: false` en `config.json`:

```bash
python3 main.py --presentation
```

### Configuración personalizada

```bash
python3 main.py --config config.json --presentation
```

### Prueba exclusiva de GPIO

```bash
python3 main.py --gpio-self-test
```

---

## ⌨️ Controles de la interfaz

| Tecla | Acción |
| :---: | :--- |
| `1` | Modo automático simulado |
| `2` | Modo mantenimiento simulado |
| `3` | Paro de emergencia simulado |
| `O` | Calibrar perfil **ojos abiertos** |
| `S` | Calibrar perfil **posible somnolencia** |
| `D` | Calibrar perfil **dormido / ojos cerrados** |
| `C` | Alias compatible de `O` |
| `L` | Mostrar u ocultar landmarks |
| `I` | Mostrar u ocultar el panel de información |
| `M` | Silenciar o reactivar el buzzer |
| `P` | Pausar o reanudar la visualización |
| `R` | Reiniciar las métricas temporales |
| `Q` / `Esc` | Cerrar la aplicación |

### Calibración de los tres estados

Realiza la calibración con el vehículo detenido, la cámara en su posición final
y una iluminación similar a la de uso:

1. Presiona `O` y permanece aproximadamente 4 segundos mirando al frente, con
   los ojos abiertos y una postura normal.
2. Presiona `S` y simula posible somnolencia durante 4 segundos: párpados
   entrecerrados, expresión relajada y una inclinación ligera y natural.
3. Presiona `D` y mantén durante 4 segundos los ojos cerrados y la postura de
   cabeza que se desea reconocer como dormido.
4. Comprueba en el panel que aparezca `O:OK S:OK D:OK`. A partir de ese
   momento se muestran el perfil detectado, su confianza y su duración.

Las muestras con baja calidad se descartan. Los valores EAR deben quedar
separados en el orden abierto > somnoliento > dormido; si se superponen, el
sistema rechaza el conjunto para evitar un clasificador ambiguo.

**dlib no se reentrena:** continúa extrayendo los 68 landmarks. El clasificador
de sesión combina **EAR, MAR, pitch, yaw, roll y mirada** para comparar cada
frame con los tres perfiles personales. La calibración se mantiene en memoria,
por lo que debe repetirse al reiniciar el programa.

---

## ⚙️ Configuración principal

El archivo [config.json](config.json) centraliza la configuración de cámara,
dlib, fatiga, GPIO, buzzer, LEDs, switch, interfaz y logging.

Para habilitar hardware real:

```json
{
  "gpio": {
    "enabled": true,
    "simulation_mode": false,
    "numbering": "BOARD"
  },
  "buzzer": {
    "enabled": true,
    "type": "pwm_native",
    "board_pin": 33,
    "active_high": false,
    "pwm_duty_cycle": 50,
    "pwm_chip": 0,
    "pwm_channel": 2,
    "min_frequency": 2000,
    "max_frequency": 5000
  }
}
```

Consulta [DOCUMENTACION_PROYECTO.md](DOCUMENTACION_PROYECTO.md) antes de
modificar umbrales temporales o parámetros de rendimiento.

---

## 🚨 Niveles de alerta

| Estado | Indicador | Buzzer |
| :--- | :--- | :--- |
| `NORMAL` / `PARPADEO` | LED verde | Apagado |
| `POSIBLE_SOMNOLENCIA` | LED amarillo intermitente | 2500 Hz, 3 pulsos |
| `ALERTA` | LED rojo intermitente | 3500 Hz, patrón recurrente |
| `ALERTA_CRITICA` | LED rojo rápido | 4500 Hz, patrón recurrente rápido |
| `MANTENIMIENTO` | LED amarillo fijo | Apagado |
| `PARO_EMERGENCIA` | LED rojo fijo | Apagado |

---

## ✅ Pruebas y validación

Ejecuta las pruebas unitarias:

```bash
python3 -m unittest discover -s tests -v
```

Valida sintaxis Python y configuración JSON:

```bash
python3 -m py_compile main.py src/*.py tests/*.py
python3 -m json.tool config.json >/dev/null
```

Las pruebas GPIO unitarias utilizan un sysfs temporal y no activan el buzzer
físico. Para validar el hardware utiliza `--gpio-self-test`.

Para medir exclusivamente cámara y visión, sin interfaz ni GPIO:

```bash
python3 scripts/benchmark_pipeline.py --seconds 15
```

---

## 📁 Estructura del proyecto

```text
tt2_drowsiness_jetson_v2/
├── main.py                         # Punto de entrada y argumentos CLI
├── config.json                     # Configuración principal
├── DOCUMENTACION_PROYECTO.md       # Documentación técnica completa
├── models/                         # Predictor dlib local, no versionado
├── scripts/
│   ├── benchmark_pipeline.py       # Benchmark reproducible cámara + visión
│   └── configure_pwm2_jetson_nano.sh
├── src/
│   ├── application.py              # Orquestación de la aplicación
│   ├── camera.py                   # Captura CSI con GStreamer
│   ├── face_analyzer.py            # Rostro, landmarks y métricas
│   ├── fatigue_detector.py         # Máquina de estados de fatiga
│   ├── alert_controller.py         # Patrones visuales y auditivos
│   ├── gpio_controller.py          # GPIO, PWM nativo y simulación
│   ├── mode_controller.py          # Modos de operación
│   ├── presentation_ui.py          # Interfaz OpenCV
│   └── event_logger.py             # Eventos CSV
├── systemd/
│   └── tt2-pwm2-pinmux.service
└── tests/
    ├── test_calibration.py
    ├── test_gpio_controller.py
    └── test_performance.py
```

---

## 🩺 Solución de problemas

### La cámara no abre

- Revisa el cable CSI y su orientación.
- Confirma que OpenCV tenga soporte GStreamer.
- Verifica `sensor_id` y `flip_method` en `config.json`.
- Comprueba que `nvarguscamerasrc` funcione fuera de la aplicación.

### No se encuentra el predictor facial

```bash
ls -lh models/shape_predictor_68_face_landmarks.dat
```

Si cambias su ubicación, actualiza `dlib.predictor_path` en `config.json`.

### El buzzer solo hace “tic”

- Confirma que Jetson-IO tenga seleccionado **`pwm2 (33)`**.
- Instala y habilita `tt2-pwm2-pinmux.service`.
- Ejecuta `python3 main.py --gpio-self-test`.

### El buzzer no se apaga

El módulo es low-trigger. No uses `PWM.stop()` sobre BOARD 33; el estado de
reposo debe ser **HIGH**, implementado como PWM habilitado al 100 %.

### Hay falsas alertas

- Completa los tres perfiles con `O`, `S` y `D` hasta ver
  `O:OK S:OK D:OK`.
- Mejora la iluminación y el encuadre.
- Revisa `ear_threshold`, PERCLOS y los umbrales temporales.

---

## 🤝 Contribuciones

Las mejoras son bienvenidas. Antes de proponer cambios:

1. Crea una rama descriptiva.
2. Mantén separados el hardware real y los backends simulados.
3. Añade o actualiza pruebas cuando cambies GPIO/PWM.
4. Ejecuta la validación completa antes de abrir un pull request.

---

## 📄 Licencia

Este repositorio todavía no incluye una licencia de código abierto. Hasta que
se añada una, todos los derechos permanecen reservados por el autor.
