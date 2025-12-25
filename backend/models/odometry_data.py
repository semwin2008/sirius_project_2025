from pydantic import BaseModel

from backend.models.quaternion import Quaternion
from backend.models.vector3 import Vector3


class OdometryData(BaseModel):
    frame_id: str
    timestamp: int
    position: Vector3
    orientation: Quaternion
    linear_velocity: Vector3
    angular_velocity: Vector3


class OdometryDataFromBin(BaseModel):
    frame_id: str
    child_frame_id: str = ""
    timestamp: int  # или str, но лучше int/float

    position_x: float
    position_y: float
    position_z: float = 0.0  # если Z не обязательный

    orientation_x: float = 0.0
    orientation_y: float = 0.0
    orientation_z: float
    orientation_w: float

    linear_velocity_x: float
    linear_velocity_y: float = 0.0
    linear_velocity_z: float = 0.0

    angular_velocity_x: float = 0.0
    angular_velocity_y: float = 0.0
    angular_velocity_z: float
