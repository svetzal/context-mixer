"""
Tests for Git gateway commit functionality.
"""
from pathlib import Path
from typing import Any

import pytest

from prompt_mixer.gateways.git import GitGateway
from prompt_mixer.domain.commit_message import CommitMessage


@pytest.fixture
def mock_subprocess(mocker):
    return mocker.patch('prompt_mixer.gateways.git.subprocess.Popen')


@pytest.fixture
def subject():
    return GitGateway()


@pytest.fixture
def repo_path(tmp_path):
    return tmp_path / "test-repo"


class DescribeGitGateway:
    
    class DescribeGetDiff:
        
        def should_get_working_directory_diff_when_staged_false(self, subject, mock_subprocess, repo_path):
            mock_process = mock_subprocess.return_value
            mock_process.communicate.return_value = ("diff output", "")
            mock_process.returncode = 0
            
            diff_output, error = subject.get_diff(repo_path, staged=False)
            
            assert diff_output == "diff output"
            assert error == ""
            mock_subprocess.assert_called_once_with(
                ["git", "diff"],
                stdout=-1,
                stderr=-1,
                text=True,
                cwd=repo_path
            )
        
        def should_get_staged_diff_when_staged_true(self, subject, mock_subprocess, repo_path):
            mock_process = mock_subprocess.return_value
            mock_process.communicate.return_value = ("staged diff output", "")
            mock_process.returncode = 0
            
            diff_output, error = subject.get_diff(repo_path, staged=True)
            
            assert diff_output == "staged diff output"
            assert error == ""
            mock_subprocess.assert_called_once_with(
                ["git", "diff", "--staged"],
                stdout=-1,
                stderr=-1,
                text=True,
                cwd=repo_path
            )
        
        def should_return_error_when_diff_fails(self, subject, mock_subprocess, repo_path):
            mock_process = mock_subprocess.return_value
            mock_process.communicate.return_value = ("", "fatal: not a git repository")
            mock_process.returncode = 128
            
            diff_output, error = subject.get_diff(repo_path, staged=False)
            
            assert diff_output == ""
            assert error == "fatal: not a git repository"
    
    class DescribeAddAll:
        
        def should_add_all_changes_successfully(self, subject, mock_subprocess, repo_path):
            mock_process = mock_subprocess.return_value
            mock_process.communicate.return_value = ("", "")
            mock_process.returncode = 0
            
            success, message = subject.add_all(repo_path)
            
            assert success is True
            assert message == "Successfully added all changes to staging area"
            mock_subprocess.assert_called_once_with(
                ["git", "add", "."],
                stdout=-1,
                stderr=-1,
                text=True,
                cwd=repo_path
            )
        
        def should_return_error_when_add_fails(self, subject, mock_subprocess, repo_path):
            mock_process = mock_subprocess.return_value
            mock_process.communicate.return_value = ("", "fatal: pathspec '.' did not match any files")
            mock_process.returncode = 1
            
            success, message = subject.add_all(repo_path)
            
            assert success is False
            assert "Failed to add changes: fatal: pathspec '.' did not match any files" in message
    
    class DescribeCommit:
        
        def should_commit_with_short_message_only(self, subject, mock_subprocess, repo_path):
            mock_process = mock_subprocess.return_value
            mock_process.communicate.return_value = ("[main abc123] Add new feature", "")
            mock_process.returncode = 0
            
            commit_message = CommitMessage(short="Add new feature", long="")
            success, message = subject.commit(repo_path, commit_message)
            
            assert success is True
            assert message == "Successfully committed changes: Add new feature"
            mock_subprocess.assert_called_once_with(
                ["git", "commit", "-m", "Add new feature"],
                stdout=-1,
                stderr=-1,
                text=True,
                cwd=repo_path
            )
        
        def should_commit_with_short_and_long_message(self, subject, mock_subprocess, repo_path):
            mock_process = mock_subprocess.return_value
            mock_process.communicate.return_value = ("[main abc123] Add new feature", "")
            mock_process.returncode = 0
            short_msg = "Add new feature"
            long_msg = "This feature enables users to perform advanced operations with improved performance."
            expected_full_msg = f"{short_msg}\n\n{long_msg}"
            
            commit_message = CommitMessage(short=short_msg, long=long_msg)
            success, message = subject.commit(repo_path, commit_message)
            
            assert success is True
            assert message == "Successfully committed changes: Add new feature"
            mock_subprocess.assert_called_once_with(
                ["git", "commit", "-m", expected_full_msg],
                stdout=-1,
                stderr=-1,
                text=True,
                cwd=repo_path
            )
        
        def should_return_error_when_commit_fails(self, subject, mock_subprocess, repo_path):
            mock_process = mock_subprocess.return_value
            mock_process.communicate.return_value = ("", "nothing to commit, working tree clean")
            mock_process.returncode = 1
            
            commit_message = CommitMessage(short="Add feature", long="")
            success, message = subject.commit(repo_path, commit_message)
            
            assert success is False
            assert "Failed to commit changes: nothing to commit, working tree clean" in message
