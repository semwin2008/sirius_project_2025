from typing import List

from pydantic import BaseModel

from backend.models.lidar_scan import LidarScan
from backend.models.odometry_data import OdometryData


class RobotInput(BaseModel):
    batch_id: int
    start_timestamp: int
    end_timestamp: int
    duration_ms: int = None
    odometries: List[OdometryData]
    lidar_scans: List[LidarScan]
    imu_readings: List[dict] = []
    images: List[str] = []

