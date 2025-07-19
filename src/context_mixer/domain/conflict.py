from pydantic import BaseModel, Field
from typing import List, Optional

from .context import Context, ContextAnalysis


class ConflictingGuidance(BaseModel):
    """
    A model representing a piece of conflicting guidance.

    Attributes:
        content: The content of the guidance
        source: The source of the guidance (e.g., "existing" or "new")
        contexts: Contexts where this guidance applies
        confidence: Confidence in context detection (0.0-1.0)
    """
    content: str = Field(..., description="The content of the guidance", max_length=2000)
    source: str = Field(..., description="The source of the guidance (e.g., 'existing' or 'new')")
    contexts: List[Context] = Field(default_factory=list, description="Contexts where this guidance applies")
    confidence: Optional[float] = Field(None, description="Confidence in context detection (0.0-1.0)", ge=0.0, le=1.0)


class Conflict(BaseModel):
    """
    A model representing a conflict between two pieces of guidance.

    Attributes:
        description: A description of what is in conflict
        conflicting_guidance: A list of the conflicting guidance
        resolution: The resolved guidance after user consultation
        context_analysis: Analysis of why contexts might make these non-conflicting
        user_provided_contexts: Contexts provided by user during resolution
    """
    description: str = Field(..., description="A description of what is in conflict", max_length=1000)
    conflicting_guidance: List[ConflictingGuidance] = Field(..., description="A list of the conflicting guidance")
    resolution: Optional[str] = Field(None, description="The resolved guidance after user consultation")
    context_analysis: Optional[str] = Field(None, description="Analysis of why contexts might make these non-conflicting")
    user_provided_contexts: List[Context] = Field(default_factory=list, description="Contexts provided by user during resolution")

class ConflictList(BaseModel):
    list: List[Conflict]
