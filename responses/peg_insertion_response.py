from pydantic import BaseModel


class ActionResponse(BaseModel):
    offset: list[float]
    reasoning: str