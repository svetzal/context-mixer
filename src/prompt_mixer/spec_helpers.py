"""
Test helpers for prompt_mixer tests.

This module contains helper classes and functions for testing prompt_mixer.
"""

from typing import List

from prompt_mixer.gateways.llm import Message


class MessageMatcher:
    """
    A custom matcher for Message objects in tests.
    
    This class is used to verify that a message contains all the expected content parts.
    It implements __eq__ to check if the other object is a list with one Message object
    and if that Message's content contains all the expected content parts.
    """
    
    def __init__(self, expected_content_parts: List[str]):
        """
        Initialize a new MessageMatcher.
        
        Args:
            expected_content_parts: A list of strings that should be present in the message content
        """
        self.expected_content_parts = expected_content_parts

    def __eq__(self, other):
        """
        Check if the other object is a list with one Message object and if that Message's
        content contains all the expected content parts.
        
        Args:
            other: The object to compare with
            
        Returns:
            True if the other object is a list with one Message object and if that Message's
            content contains all the expected content parts, False otherwise
        """
        # Check if other is a list with one Message
        if not isinstance(other, list) or len(other) != 1:
            return False

        # Check if the message contains all expected content parts
        message = other[0]
        if not isinstance(message, Message):
            return False

        return all(part in message.content for part in self.expected_content_parts)

    def __repr__(self):
        """
        Return a string representation of the MessageMatcher.
        
        Returns:
            A string representation of the MessageMatcher
        """
        return f"<MessageMatcher: expected_content_parts={self.expected_content_parts}>"