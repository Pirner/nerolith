from dataclasses import dataclass


@dataclass
class ModelConfig:
    backbone: str
    architecture: str
    n_classes: int
    im_height: int
    im_width: int
