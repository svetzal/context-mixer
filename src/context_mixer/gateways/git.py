"""
Git operations gateway for Context Mixer.

This module provides a GitGateway class that handles all git operations
by delegating to the system git command. It serves as a boundary between
our logic and actual git operations, making it easier to mock for testing.
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from context_mixer.domain.commit_message import CommitMessage


class GitGateway:
    """
    Gateway for Git operations.

    This class handles all git operations by delegating to the system git command.
    It serves as a boundary between our logic and actual git operations,
    making it easier to mock for testing.
    """

    def run_git_command(self, args: List[str], cwd: Optional[Path] = None) -> Tuple[str, str, int]:
        """
        Run a git command with the given arguments.

        Args:
            args: List of command arguments (excluding 'git')
            cwd: Working directory for the command

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        cmd = ["git"] + args
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        )
        stdout, stderr = process.communicate()
        return stdout, stderr, process.returncode

    def clone(self, repo_url: str, target_path: Path) -> Tuple[bool, str]:
        """
        Clone a git repository.

        Args:
            repo_url: URL of the repository to clone
            target_path: Path where the repository should be cloned

        Returns:
            Tuple of (success, message)
        """
        stdout, stderr, return_code = self.run_git_command(["clone", repo_url, str(target_path)])

        if return_code == 0:
            return True, f"Successfully cloned {repo_url} to {target_path}"
        else:
            return False, f"Failed to clone repository: {stderr}"

    def init(self, path: Path) -> Tuple[bool, str]:
        """
        Initialize a new git repository.

        Args:
            path: Path where the repository should be initialized

        Returns:
            Tuple of (success, message)
        """
        stdout, stderr, return_code = self.run_git_command(["init"], cwd=path)

        if return_code == 0:
            return True, f"Successfully initialized git repository at {path}"
        else:
            return False, f"Failed to initialize repository: {stderr}"

    def add_remote(self, path: Path, name: str, url: str) -> Tuple[bool, str]:
        """
        Add a remote to a git repository.

        Args:
            path: Path to the git repository
            name: Name of the remote
            url: URL of the remote

        Returns:
            Tuple of (success, message)
        """
        stdout, stderr, return_code = self.run_git_command(
            ["remote", "add", name, url],
            cwd=path
        )

        if return_code == 0:
            return True, f"Successfully added remote {name} ({url})"
        else:
            return False, f"Failed to add remote: {stderr}"

    def get_diff(self, path: Path, staged: bool = False) -> Tuple[str, str]:
        """
        Get the git diff for the repository.

        Args:
            path: Path to the git repository
            staged: If True, get staged changes; if False, get working directory changes

        Returns:
            Tuple of (diff_output, error_message)
        """
        args = ["diff"]
        if staged:
            args.append("--staged")

        stdout, stderr, return_code = self.run_git_command(args, cwd=path)

        if return_code == 0:
            return stdout, ""
        else:
            return "", stderr

    def add_all(self, path: Path) -> Tuple[bool, str]:
        """
        Add all changes to the staging area.

        Args:
            path: Path to the git repository

        Returns:
            Tuple of (success, message)
        """
        stdout, stderr, return_code = self.run_git_command(["add", "."], cwd=path)

        if return_code == 0:
            return True, "Successfully added all changes to staging area"
        else:
            return False, f"Failed to add changes: {stderr}"

    def commit(self, path: Path, commit_message: CommitMessage) -> Tuple[bool, str]:
        """
        Commit changes with a commit message.

        Args:
            path: Path to the git repository
            commit_message: CommitMessage object containing short and long message parts

        Returns:
            Tuple of (success, message)
        """
        # Get formatted message from the CommitMessage object
        full_message = commit_message.format_message()

        stdout, stderr, return_code = self.run_git_command(
            ["commit", "-m", full_message],
            cwd=path
        )

        if return_code == 0:
            return True, f"Successfully committed changes: {commit_message.short}"
        else:
            return False, f"Failed to commit changes: {stderr}"
