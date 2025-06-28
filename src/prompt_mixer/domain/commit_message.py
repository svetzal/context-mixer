"""
Domain models for git commit messages.
"""

from pydantic import BaseModel, Field


class CommitMessage(BaseModel):
    """
    Model for a git commit message with short and long forms.

    This model represents a structured commit message that includes
    both a short summary line and a longer detailed description.
    """

    short: str = Field(
        description="Short commit message (50 characters or less)",
        max_length=50
    )

    long: str = Field(
        description="Longer commit message with detailed explanation"
    )

    def format_message(self) -> str:
        """
        Format the commit message for git.

        Returns:
            Formatted commit message with short and long parts properly combined
        """
        if self.long.strip():
            return f"{self.short}\n\n{self.long}"
        else:
            return self.short
