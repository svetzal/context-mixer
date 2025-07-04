"""
Git commit operations using LLM-generated commit messages.

This module provides high-level operations for committing changes to git
repositories with automatically generated commit messages.
"""

from pathlib import Path
from typing import Tuple

from context_mixer.gateways.git import GitGateway
from context_mixer.gateways.llm import LLMGateway
from context_mixer.domain.commit_message import CommitMessage
from context_mixer.commands.operations.commit_message import generate_commit_message


class CommitOperation:
    """
    High-level operation for committing changes with LLM-generated messages.

    This class combines the GitGateway and LLMGateway to provide a complete
    workflow for committing changes with automatically generated commit messages.
    """

    def __init__(self, git_gateway: GitGateway, llm_gateway: LLMGateway):
        """
        Initialize the commit operation.

        Args:
            git_gateway: Gateway for git operations
            llm_gateway: Gateway for LLM operations
        """
        self.git_gateway = git_gateway
        self.llm_gateway = llm_gateway

    def commit_changes(self, repo_path: Path, stage_all: bool = True) -> Tuple[bool, str, CommitMessage]:
        """
        Commit changes with an LLM-generated commit message.

        This method performs the following steps:
        1. Optionally stage all changes
        2. Get the diff of staged changes
        3. Generate a commit message using the LLM
        4. Commit the changes with the generated message

        Args:
            repo_path: Path to the git repository
            stage_all: Whether to stage all changes before committing

        Returns:
            Tuple of (success, message, commit_message_used)
        """
        # Stage all changes if requested
        if stage_all:
            success, message = self.git_gateway.add_all(repo_path)
            if not success:
                return False, f"Failed to stage changes: {message}", CommitMessage(short="", long="")

        # Get the diff of staged changes
        diff_output, error = self.git_gateway.get_diff(repo_path, staged=True)
        if error:
            return False, f"Failed to get diff: {error}", CommitMessage(short="", long="")

        # Check if there are any changes to commit
        if not diff_output.strip():
            return False, "No staged changes to commit", CommitMessage(short="", long="")

        # Generate commit message using LLM
        try:
            commit_message = generate_commit_message(self.llm_gateway, diff_output)
        except Exception as e:
            return False, f"Failed to generate commit message: {str(e)}", CommitMessage(short="", long="")

        # Commit the changes
        success, message = self.git_gateway.commit(repo_path, commit_message)

        if success:
            return True, message, commit_message
        else:
            return False, message, commit_message

    def preview_commit_message(self, repo_path: Path, stage_all: bool = True) -> Tuple[bool, str, CommitMessage]:
        """
        Preview the commit message that would be generated without committing.

        Args:
            repo_path: Path to the git repository
            stage_all: Whether to consider all changes (not just staged)

        Returns:
            Tuple of (success, message, generated_commit_message)
        """
        # Get the appropriate diff
        if stage_all:
            diff_output, error = self.git_gateway.get_diff(repo_path, staged=False)
        else:
            diff_output, error = self.git_gateway.get_diff(repo_path, staged=True)

        if error:
            return False, f"Failed to get diff: {error}", CommitMessage(short="", long="")

        if not diff_output.strip():
            message = "No changes to preview" if stage_all else "No staged changes to preview"
            return False, message, CommitMessage(short="", long="")

        # Generate commit message using LLM
        try:
            commit_message = generate_commit_message(self.llm_gateway, diff_output)
            return True, "Commit message generated successfully", commit_message
        except Exception as e:
            return False, f"Failed to generate commit message: {str(e)}", CommitMessage(short="", long="")
