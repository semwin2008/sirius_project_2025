from pydantic import BaseModel
from typing import List


class LidarScan(BaseModel):
    frame_id: str
    timestamp: int
    angle_min: float
    angle_max: float
    range_min: float
    range_max: float
    ranges: List[float]