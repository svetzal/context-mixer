"""
Operations for generating commit messages from diffs.

This module contains business logic for analyzing git diffs and generating
appropriate commit messages using LLM capabilities.
"""

from mojentic.llm import LLMMessage, MessageRole

from context_mixer.domain.commit_message import CommitMessage
from context_mixer.domain.llm_instructions import git_commit_system_message
from context_mixer.gateways.llm import LLMGateway


def generate_commit_message(llm_gateway: LLMGateway, diff: str) -> CommitMessage:
    """
    Generate a commit message from a git diff using an LLM.

    This function analyzes a git diff and generates an appropriate commit message
    following conventional commit message guidelines.

    Args:
        llm_gateway: The LLM gateway to use for generation
        diff: The git diff to analyze

    Returns:
        A CommitMessage object with short and long forms

    Raises:
        Exception: If the LLM fails to generate a valid commit message
    """

    messages = [
        git_commit_system_message,
        LLMMessage(
            content=f"Generate a commit message for this diff:\n\n{diff}"
        )
    ]

    return llm_gateway.generate_object(messages, CommitMessage)
