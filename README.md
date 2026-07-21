# tt2_drowsiness_jetson

Sistema de deteccion de somnolencia para NVIDIA Jetson Nano con camara CSI,
OpenCV, dlib y alertas mediante GPIO.

La instalacion, configuracion, conexiones y diagnostico del buzzer se describen
en [DOCUMENTACION_PROYECTO.md](DOCUMENTACION_PROYECTO.md).

El modelo local `models/shape_predictor_68_face_landmarks.dat` no se almacena
en Git debido a su tamano (aproximadamente 96 MB). Debe colocarse en esa ruta
antes de ejecutar la aplicacion.
