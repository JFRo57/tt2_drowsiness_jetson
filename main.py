#!/usr/bin/env python3
import argparse
import json
import os
import sys

from src.application import DrowsinessApplication, default_config, deep_update
from src.gpio_controller import GPIOController


def load_config(path):
    cfg = default_config()
    if os.path.exists(path):
        with open(path, "r") as fh:
            loaded = json.load(fh)
        deep_update(cfg, loaded)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Sistema de deteccion de somnolencia en Jetson Nano")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--presentation", action="store_true", help="Mostrar interfaz OpenCV de presentacion")
    mode.add_argument("--headless", action="store_true", help="Ejecutar sin interfaz")
    parser.add_argument("--state_machine", action="store_true", help="Mostrar otra ventana con la maquina de estados en tiempo real")
    parser.add_argument("--simulation", action="store_true", help="Forzar GPIO, buzzer, LEDs y switch simulados")
    parser.add_argument("--gpio-self-test", action="store_true", help="Probar LEDs y buzzer fisicos sin camara ni interfaz")
    parser.add_argument("--config", default="config.json", help="Ruta al archivo JSON de configuracion")
    return parser.parse_args()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    args = parse_args()
    cfg = load_config(args.config)
    if args.headless:
        cfg["interface"]["presentation_enabled"] = False
    if args.presentation:
        cfg["interface"]["presentation_enabled"] = True
    if args.state_machine:
        cfg["interface"]["presentation_enabled"] = True
        cfg["interface"]["state_machine_enabled"] = True
    if args.simulation:
        cfg["gpio"]["enabled"] = False
        cfg["gpio"]["simulation_mode"] = True
        cfg["switch"]["enabled"] = False
    if args.gpio_self_test:
        cfg["gpio"]["enabled"] = True
        cfg["gpio"]["simulation_mode"] = False
        cfg["switch"]["enabled"] = False
        return GPIOController(cfg, force_simulation=False).self_test()
    app = DrowsinessApplication(cfg, simulation=args.simulation)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
