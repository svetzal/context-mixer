from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

from rich.console import Console
from context_mixer.config import Config
from context_mixer.gateways.llm import LLMGateway
from context_mixer.gateways.git import GitGateway
from context_mixer.domain.knowledge_store import KnowledgeStore


@dataclass
class CommandContext:
    """
    Shared context for command execution.
    
    Contains all the common dependencies and state that commands need.
    """
    console: Console
    config: Config
    llm_gateway: Optional[LLMGateway] = None
    git_gateway: Optional[GitGateway] = None
    knowledge_store: Optional[KnowledgeStore] = None
    
    # Command-specific parameters can be added as needed
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class CommandResult:
    """
    Result of command execution.
    
    Provides a standardized way to return success/failure status and data.
    """
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


class Command(ABC):
    """
    Abstract base class for all commands.
    
    Implements the Command pattern as specified in the architectural improvements backlog.
    All CLI commands should inherit from this class and implement the execute method.
    """
    
    @abstractmethod
    async def execute(self, context: CommandContext) -> CommandResult:
        """
        Execute the command with the given context.
        
        Args:
            context: CommandContext containing all necessary dependencies and parameters
            
        Returns:
            CommandResult indicating success/failure and any relevant data
        """
        pass