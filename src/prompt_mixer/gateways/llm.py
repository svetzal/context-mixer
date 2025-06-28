"""
Gateway for LLM interactions.

This module provides a gateway for interacting with LLMs, isolating the mojentic library
from the rest of the application.
"""

from typing import List, Optional

from mojentic.llm import LLMBroker, LLMMessage


class Message:
    """
    A message to be sent to an LLM.
    
    This class wraps the mojentic LLMMessage to isolate it from the rest of the application.
    """
    
    def __init__(self, content: str):
        """
        Initialize a new Message.
        
        Args:
            content: The content of the message
        """
        self.content = content


class LLMGateway:
    """
    Gateway for interacting with LLMs.
    
    This class provides a gateway for interacting with LLMs, isolating the mojentic library
    from the rest of the application.
    """
    
    def __init__(self, model: str, gateway: Optional[object] = None):
        """
        Initialize a new LLMGateway.
        
        Args:
            model: The model to use for generating content
            gateway: The gateway to use for interacting with the LLM provider
        """
        self.llm_broker = LLMBroker(model=model, gateway=gateway)
    
    def generate(self, messages: List[Message]) -> str:
        """
        Generate content using the LLM.
        
        Args:
            messages: A list of messages to send to the LLM
            
        Returns:
            The generated content
        """
        # Convert our Message objects to mojentic LLMMessage objects
        llm_messages = [LLMMessage(content=message.content) for message in messages]
        
        # Generate content using the mojentic LLMBroker
        return self.llm_broker.generate(messages=llm_messages)