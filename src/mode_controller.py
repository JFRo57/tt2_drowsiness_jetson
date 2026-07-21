class ModeController(object):
    AUTOMATIC = "AUTOMATIC"
    MAINTENANCE = "MAINTENANCE"
    EMERGENCY = "EMERGENCY"

    def __init__(self):
        self.mode = self.AUTOMATIC
        self.source = "simulado"

    def set_mode(self, mode, source="simulado"):
        if mode in (self.AUTOMATIC, self.MAINTENANCE, self.EMERGENCY):
            self.mode = mode
            self.source = source

    def update_from_switch(self, gpio_controller):
        mode = gpio_controller.read_switch_mode()
        if mode:
            self.set_mode(mode, gpio_controller.mode_source)
        return self.mode
