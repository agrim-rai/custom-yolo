# minimal stub for detection-only training (no HUB usage)
HUB_WEB_ROOT = "https://hub.ultralytics.com"
PREFIX = "Ultralytics"


class HUBTrainingSession:
    """Stub; not used when training from local yaml/pt."""

    @staticmethod
    def create_session(model):
        raise RuntimeError("HUB is disabled in this build")
