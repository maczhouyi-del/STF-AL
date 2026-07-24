from __future__ import annotations

from .stfal import STFALModel


class BaselineM0(STFALModel):
    def __init__(self, config):
        config = dict(config)
        config["model"] = dict(config["model"])
        config["model"]["variant"] = "m0"
        super().__init__(config)
