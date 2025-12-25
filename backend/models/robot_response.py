from typing import List

from pydantic import BaseModel, Field


class RobotResponse(BaseModel):
    interest: int
    joke: str
