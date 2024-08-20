from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DataPoint:
    im_path: str
    annotation: List[Dict]
    image_metadata: Dict
