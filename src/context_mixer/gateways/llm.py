"""
Gateway for LLM interactions.

This module provides a gateway for interacting with LLMs, isolating the mojentic library
from the rest of the application.
"""

import json
from typing import List, Optional, Type, TypeVar

from mojentic.llm import LLMBroker, LLMMessage
from pydantic import BaseModel


T = TypeVar('T', bound=BaseModel)

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

    def generate(self, messages: List[LLMMessage]) -> str:
        """
        Generate content using the LLM.

        Args:
            messages: A list of messages to send to the LLM

        Returns:
            The generated content
        """
        # Generate content using the mojentic LLMBroker
        return self.llm_broker.generate(messages=messages)

    def generate_object(self, messages: List[LLMMessage], object_model: Type[T]) -> T:
        """
        Generate a structured object using the LLM.

        This method uses the LLM to generate a structured object based on a Pydantic model.
        It instructs the LLM to generate JSON that conforms to the model's schema, then
        parses that JSON into an instance of the model.

        Args:
            messages: A list of messages to send to the LLM
            object_model: The Pydantic model class to use for parsing the response

        Returns:
            An instance of the specified Pydantic model
        """
        return self.llm_broker.generate_object(messages=messages, object_model=object_model)
