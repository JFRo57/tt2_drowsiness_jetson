import signal


class ShutdownManager(object):
    def __init__(self):
        self.requested = False
        self.reason = ""
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        self.request("Senal %s recibida" % signum)

    def request(self, reason="Cierre solicitado"):
        self.requested = True
        self.reason = reason
