from typing import List
from pydantic import BaseModel, Field


class ModelResponseDto(BaseModel):
    interest: int = Field(description="Interest score: 0 - not interesting, 1 - interesting")
    joke: str = Field(description="Joke string")

