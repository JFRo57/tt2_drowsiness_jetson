#!/bin/sh
set -eu

BUSYBOX=/usr/lib/initramfs-tools/bin/busybox
COMPATIBLE=/proc/device-tree/compatible
PWM_CHIP=/sys/class/pwm/pwmchip0
PWM_CHANNEL=2
PWM_PATH="$PWM_CHIP/pwm$PWM_CHANNEL"
PERIOD_NS=333333

if [ "$(id -u)" -ne 0 ]; then
    echo "Este script debe ejecutarse como root." >&2
    exit 1
fi
if [ ! -x "$BUSYBOX" ]; then
    echo "No se encontro $BUSYBOX (paquete busybox-initramfs)." >&2
    exit 1
fi
if ! "$BUSYBOX" grep -aq "nvidia,tegra210" "$COMPATIBLE"; then
    echo "Plataforma no compatible: se esperaba Jetson Nano/Tegra210." >&2
    exit 1
fi

# BOARD 33: PE6 -> PWM2/SFIO. Jetson-IO declara el pinmux, pero en algunas
# instalaciones L4T 32.x estos registros no quedan aplicados tras arrancar.
"$BUSYBOX" devmem 0x70003248 32 0x46
"$BUSYBOX" devmem 0x6000d100 32 0x00

if [ ! -d "$PWM_CHIP" ]; then
    echo "No existe $PWM_CHIP." >&2
    exit 1
fi
if [ ! -d "$PWM_PATH" ]; then
    printf '%s' "$PWM_CHANNEL" > "$PWM_CHIP/export"
    tries=0
    while [ ! -d "$PWM_PATH" ] && [ "$tries" -lt 100 ]; do
        sleep 0.01
        tries=$((tries + 1))
    done
fi
if [ ! -d "$PWM_PATH" ]; then
    echo "No se pudo exportar PWM2." >&2
    exit 1
fi

# MH-FMD/YL-44 es low-level trigger: PWM habilitado al 100 % equivale a
# HIGH constante y es el estado silencioso. No deshabilitar el canal.
if [ "$(cat "$PWM_PATH/period")" != "$PERIOD_NS" ]; then
    printf '0' > "$PWM_PATH/duty_cycle"
    printf '%s' "$PERIOD_NS" > "$PWM_PATH/period"
fi
printf '%s' "$PERIOD_NS" > "$PWM_PATH/duty_cycle"
if [ "$(cat "$PWM_PATH/enable")" != "1" ]; then
    printf '1' > "$PWM_PATH/enable"
fi
