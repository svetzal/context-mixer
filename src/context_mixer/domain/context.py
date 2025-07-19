from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ContextType(str, Enum):
    """Types of contexts that can be applied to guidance rules."""
    ARCHITECTURAL = "architectural"
    TEMPORAL = "temporal"
    PLATFORM = "platform"
    ENVIRONMENT = "environment"
    LANGUAGE = "language"
    FRAMEWORK = "framework"
    TEAM = "team"
    PROJECT_PHASE = "project_phase"
    CUSTOM = "custom"


class Context(BaseModel):
    """
    Represents a context in which a rule or guidance applies.
    
    Contexts help distinguish between rules that might appear conflicting
    but actually apply to different situations, environments, or scopes.
    
    Attributes:
        type: The type of context (architectural, platform, environment, etc.)
        value: The specific value within that context type
        description: Optional human-readable description of the context
        confidence: Confidence level in context detection (0.0-1.0)
    """
    type: ContextType = Field(..., description="The type of context")
    value: str = Field(..., description="The specific value within the context type")
    description: Optional[str] = Field(None, description="Human-readable description of the context")
    confidence: Optional[float] = Field(None, description="Confidence in context detection (0.0-1.0)", ge=0.0, le=1.0)
    
    def __str__(self) -> str:
        """String representation of the context."""
        return f"{self.type.value}:{self.value}"
    
    def __eq__(self, other) -> bool:
        """Check equality based on type and value."""
        if not isinstance(other, Context):
            return False
        return self.type == other.type and self.value == other.value
    
    def __hash__(self) -> int:
        """Hash based on type and value for use in sets."""
        return hash((self.type, self.value))


class ContextPattern(BaseModel):
    """
    Represents a pattern for detecting contexts in content.
    
    This can be used to train context detectors and improve
    context detection accuracy over time.
    
    Attributes:
        pattern: The text pattern that indicates this context
        context: The context that this pattern indicates
        weight: Weight/importance of this pattern (higher = more important)
        examples: Example texts where this pattern applies
    """
    pattern: str = Field(..., description="The text pattern that indicates this context")
    context: Context = Field(..., description="The context that this pattern indicates")
    weight: float = Field(1.0, description="Weight/importance of this pattern", ge=0.0)
    examples: List[str] = Field(default_factory=list, description="Example texts where this pattern applies")


class ContextAnalysis(BaseModel):
    """
    Analysis of contexts detected in content.
    
    This provides detailed information about what contexts were found,
    how confident the detection was, and any conflicts or overlaps.
    
    Attributes:
        detected_contexts: List of contexts detected in the content
        confidence_score: Overall confidence in the context analysis
        analysis_notes: Human-readable notes about the analysis
        potential_conflicts: Contexts that might conflict with each other
    """
    detected_contexts: List[Context] = Field(default_factory=list, description="List of contexts detected")
    confidence_score: float = Field(0.0, description="Overall confidence in the analysis", ge=0.0, le=1.0)
    analysis_notes: Optional[str] = Field(None, description="Human-readable notes about the analysis")
    potential_conflicts: List[str] = Field(default_factory=list, description="Notes about potential context conflicts")