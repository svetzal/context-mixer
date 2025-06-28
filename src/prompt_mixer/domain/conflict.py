from pydantic import BaseModel, Field
from typing import List, Optional


class ConflictingGuidance(BaseModel):
    """
    A model representing a piece of conflicting guidance.

    Attributes:
        content: The content of the guidance
        source: The source of the guidance (e.g., "existing" or "new")
    """
    content: str = Field(..., description="The content of the guidance", max_length=250)
    source: str = Field(..., description="The source of the guidance (e.g., 'existing' or 'new')")


class Conflict(BaseModel):
    """
    A model representing a conflict between two pieces of guidance.

    Attributes:
        description: A description of what is in conflict
        conflicting_guidance: A list of the conflicting guidance
        resolution: The resolved guidance after user consultation
    """
    description: str = Field(..., description="A description of what is in conflict", max_length=250)
    conflicting_guidance: List[ConflictingGuidance] = Field(..., description="A list of the conflicting guidance")
    resolution: Optional[str] = Field(None, description="The resolved guidance after user consultation")

class ConflictList(BaseModel):
    list: List[Conflict]
