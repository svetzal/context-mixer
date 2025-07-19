"""
Gateway for LLM interactions.

This module provides a gateway for interacting with LLMs, isolating the mojentic library
from the rest of the application.
"""

import json
from enum import Enum
from typing import List, Optional, Type, TypeVar, Dict, Any, Union
from datetime import datetime

from mojentic.llm import LLMBroker, LLMMessage
from pydantic import BaseModel, Field


T = TypeVar('T', bound=BaseModel)


class LLMCall(BaseModel):
    """
    Represents a logged call to an LLM method.

    This class captures the details of each call made to the LLM gateway,
    including the method name, arguments, timestamp, and response.
    """
    method: str
    args: tuple
    kwargs: dict
    timestamp: datetime = Field(default_factory=datetime.now)
    response: Any = None

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

    def generate_embeddings(self, content: str) -> List[float]:
        """
        Generate numerical embeddings for the given content using the LLM.

        Args:
            content: The text content to generate embeddings for

        Returns:
            A list of floating point numbers representing the content embedding vector
        """
        return self.llm_broker.adapter.calculate_embeddings(content)


class MockLLMGateway:
    """
    Mock implementation of LLMGateway for testing purposes.

    This class provides a completely independent mock implementation that logs all calls 
    and allows configurable responses for different scenarios. It's designed to make
    testing LLM interactions easier and more reliable without any coupling to the real
    LLMGateway implementation.
    """

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        """
        Initialize a new MockLLMGateway.

        Args:
            responses: A dictionary mapping method names to their responses.
                      Can contain default responses or scenario-specific responses.
        """
        # Don't call super().__init__() to avoid creating a real LLMBroker
        self.responses = responses or {}
        self.call_log: List[LLMCall] = []

    def _log_call(self, method: str, *args, **kwargs) -> LLMCall:
        """
        Log a method call and return the call record.

        Args:
            method: The name of the method being called
            *args: Positional arguments passed to the method
            **kwargs: Keyword arguments passed to the method

        Returns:
            The logged call record
        """
        call = LLMCall(
            method=method,
            args=args,
            kwargs=kwargs,
            timestamp=datetime.now()
        )
        self.call_log.append(call)
        return call

    def _get_response(self, method: str, *args, **kwargs) -> Any:
        """
        Get the configured response for a method call.

        Args:
            method: The name of the method being called
            *args: Positional arguments passed to the method
            **kwargs: Keyword arguments passed to the method

        Returns:
            The configured response for this method call
        """
        # Check for scenario-specific responses first
        scenario_key = f"{method}_scenario"
        if scenario_key in self.responses:
            scenarios = self.responses[scenario_key]
            # Use the first scenario if multiple are available
            if isinstance(scenarios, list) and scenarios:
                return scenarios.pop(0)
            elif not isinstance(scenarios, list):
                return scenarios

        # Check for method-specific default response
        if method in self.responses:
            response = self.responses[method]
            # Special handling: if the response is a list and we're dealing with generate method,
            # treat it as a sequence of responses. For other methods like generate_embeddings,
            # the list itself might be the intended response.
            if isinstance(response, list):
                if method == 'generate':
                    if response:
                        # For generate method, lists are sequences of responses
                        return response.pop(0)
                    # If list is empty, fall through to default handling
                else:
                    # For other methods, return the list as-is (e.g., embeddings)
                    return response
            else:
                return response

        # Return sensible defaults based on method (don't check for 'default' fallback here)
        if method == 'generate':
            return "Mock LLM response"
        elif method == 'generate_object':
            # This is tricky - we need to return an instance of the requested model
            # For now, return None and let the test handle it
            return None
        elif method == 'generate_embeddings':
            return [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding vector
        else:
            # Only check for default fallback for unknown methods
            if 'default' in self.responses:
                return self.responses['default']
            return None

    def generate(self, messages: List[LLMMessage]) -> str:
        """
        Mock implementation of generate method.

        Args:
            messages: A list of messages to send to the LLM

        Returns:
            The configured mock response
        """
        call = self._log_call('generate', messages)
        response = self._get_response('generate', messages)
        call.response = response
        return response

    def generate_object(self, messages: List[LLMMessage], object_model: Type[T]) -> T:
        """
        Mock implementation of generate_object method.

        Args:
            messages: A list of messages to send to the LLM
            object_model: The Pydantic model class to use for parsing the response

        Returns:
            The configured mock response (should be an instance of object_model)
        """
        call = self._log_call('generate_object', messages, object_model)
        response = self._get_response('generate_object', messages, object_model)
        call.response = response

        # If no specific response is configured, try to create a default instance
        if response is None:
            try:
                # Try to create a default instance with empty/default values
                response = object_model()
            except Exception:
                # If that fails, the test will need to provide a proper response
                raise ValueError(f"MockLLMGateway: No response configured for generate_object with {object_model.__name__}. "
                               f"Please configure a response in the 'generate_object' key of the responses dict.")

        call.response = response
        return response

    def generate_embeddings(self, content: str) -> List[float]:
        """
        Mock implementation of generate_embeddings method.

        Args:
            content: The text content to generate embeddings for

        Returns:
            The configured mock embedding vector
        """
        call = self._log_call('generate_embeddings', content)
        response = self._get_response('generate_embeddings', content)
        call.response = response
        return response

    def get_calls(self, method: Optional[str] = None) -> List[LLMCall]:
        """
        Get all logged calls, optionally filtered by method name.

        Args:
            method: Optional method name to filter by

        Returns:
            List of logged calls, optionally filtered by method
        """
        if method is None:
            return self.call_log.copy()
        return [call for call in self.call_log if call.method == method]

    def get_call_count(self, method: Optional[str] = None) -> int:
        """
        Get the number of logged calls, optionally filtered by method name.

        Args:
            method: Optional method name to filter by

        Returns:
            Number of logged calls, optionally filtered by method
        """
        return len(self.get_calls(method))

    def clear_calls(self):
        """Clear all logged calls."""
        self.call_log.clear()

    def was_called(self, method: str) -> bool:
        """
        Check if a method was called.

        Args:
            method: The method name to check

        Returns:
            True if the method was called, False otherwise
        """
        return any(call.method == method for call in self.call_log)

    def assert_called_with(self, method: str, *args, **kwargs) -> bool:
        """
        Assert that a method was called with specific arguments.

        Args:
            method: The method name to check
            *args: Expected positional arguments
            **kwargs: Expected keyword arguments

        Returns:
            True if the method was called with the specified arguments
        """
        for call in self.call_log:
            if (call.method == method and 
                call.args == args and 
                call.kwargs == kwargs):
                return True
        return False


class OpenAIModels(str, Enum):
    """Available OpenAI models."""
    O4_MINI = "o4-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
