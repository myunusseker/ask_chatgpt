from pydantic import BaseModel


class ActionResponse(BaseModel):
    force: list[float]
    reasoning: str