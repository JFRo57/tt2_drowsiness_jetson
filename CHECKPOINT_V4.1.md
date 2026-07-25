# Checkpoint 4.1

Fecha: 24 de julio de 2026  
Rama: `v4`

## Estado de esta versión

La versión 4.1 consolida el nuevo flujo de calibración personalizada para el
sistema de detección de fatiga y somnolencia en Jetson. La calibración se
ejecuta como un proceso independiente: mientras está activa, el detector
principal no calcula fatiga, PERCLOS ni eventos de somnolencia.

Si no existe un perfil válido, el sistema informa al usuario antes de iniciar:

- Click, `Espacio` o `Enter` inicia la calibración.
- `N` omite la calibración únicamente durante la sesión actual y activa los
  parámetros de respaldo.
- La omisión no se guarda como una calibración válida.
- Al completar la calibración se activa automáticamente el flujo normal de
  monitoreo.

## Flujo de calibración

1. Ojos normalmente abiertos.
2. Apertura ocular reducida.
3. Ojos completamente cerrados.
4. Observación natural de parpadeos.
5. Parpadeos voluntarios, cuando la observación natural no aporta suficientes
   eventos válidos.
6. Guardado del perfil y transición al detector principal.

Cada etapa requiere confirmación del usuario. Si una etapa falla, solamente se
repite esa etapa y se muestra la causa del rechazo.

## Cambios incluidos

- La duración de captura comienza después del periodo de preparación.
- Los eventos de teclado y mouse se procesan continuamente durante la
  calibración.
- Mensajes, progreso, estados y diagnósticos de rechazo visibles en pantalla.
- Conteo de rechazos por rostro, calidad ocular, EAR y seguimiento.
- Validación de pose menos estricta, apropiada para cámaras instaladas en
  diferentes posiciones dentro de distintos vehículos.
- Conservación temporal del rostro durante ojos cerrados y parpadeos para
  evitar perder el seguimiento al cerrar los párpados.
- Tratamiento robusto de las referencias de ojos abiertos, reducidos y
  cerrados, incluyendo ajuste de la referencia reducida cuando los extremos
  válidos son distinguibles.
- Umbrales específicos y más sensibles para detectar parpadeos durante la
  calibración dinámica.
- Telemetría dinámica en pantalla: cantidad de parpadeos, estado, nivel de
  cierre y cierre máximo.
- Corrección del cierre de la aplicación al renderizar la fase de observación
  natural (`NameError: self is not defined`).
- Nuevas pruebas de regresión para el aviso inicial, omisión temporal,
  referencias oculares variables, detección dinámica y renderizado de
  telemetría.

## Validación

- Suite automatizada: 76 pruebas aprobadas.
- Revisión de diferencias Git sin errores de espacios o formato.
- Prueba de hardware pendiente de confirmación final en el vehículo, en
  especial para completar la observación de parpadeos naturales y verificar el
  perfil persistido.

## Ejecución

```bash
python3 main.py --presentation --simulation
```

Este documento identifica el estado exacto publicado como checkpoint 4.1.
