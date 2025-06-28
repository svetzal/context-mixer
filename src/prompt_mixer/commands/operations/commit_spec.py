"""
Tests for commit operations.
"""

import pytest

from prompt_mixer.commands.operations.commit import CommitOperation
from prompt_mixer.gateways.git import GitGateway
from prompt_mixer.gateways.llm import LLMGateway
from prompt_mixer.domain.commit_message import CommitMessage


@pytest.fixture
def mock_git_gateway(mocker):
    return mocker.MagicMock(spec=GitGateway)


@pytest.fixture
def mock_llm_gateway(mocker):
    return mocker.MagicMock(spec=LLMGateway)


@pytest.fixture
def mock_generate_commit_message(mocker):
    return mocker.patch('prompt_mixer.commands.operations.commit.generate_commit_message')


@pytest.fixture
def subject(mock_git_gateway, mock_llm_gateway):
    return CommitOperation(mock_git_gateway, mock_llm_gateway)


@pytest.fixture
def repo_path(tmp_path):
    return tmp_path / "test-repo"


@pytest.fixture
def sample_commit_message():
    return CommitMessage(
        short="Add new feature",
        long="This commit adds a new feature that enhances user experience with improved functionality."
    )


class DescribeCommitOperation:
    
    class DescribeCommitChanges:
        
        def should_commit_changes_with_staging(self, subject, mock_git_gateway, mock_llm_gateway, mock_generate_commit_message, repo_path, sample_commit_message):
            mock_git_gateway.add_all.return_value = (True, "Successfully staged")
            mock_git_gateway.get_diff.return_value = ("sample diff output", "")
            mock_generate_commit_message.return_value = sample_commit_message
            mock_git_gateway.commit.return_value = (True, "Successfully committed")
            
            success, message, commit_msg = subject.commit_changes(repo_path, stage_all=True)
            
            assert success is True
            assert message == "Successfully committed"
            assert commit_msg == sample_commit_message
            mock_git_gateway.add_all.assert_called_once_with(repo_path)
            mock_git_gateway.get_diff.assert_called_once_with(repo_path, staged=True)
            mock_generate_commit_message.assert_called_once_with(mock_llm_gateway, "sample diff output")
            mock_git_gateway.commit.assert_called_once_with(repo_path, sample_commit_message)
        
        def should_commit_changes_without_staging(self, subject, mock_git_gateway, mock_llm_gateway, mock_generate_commit_message, repo_path, sample_commit_message):
            mock_git_gateway.get_diff.return_value = ("sample diff output", "")
            mock_generate_commit_message.return_value = sample_commit_message
            mock_git_gateway.commit.return_value = (True, "Successfully committed")
            
            success, message, commit_msg = subject.commit_changes(repo_path, stage_all=False)
            
            assert success is True
            assert message == "Successfully committed"
            assert commit_msg == sample_commit_message
            mock_git_gateway.add_all.assert_not_called()
            mock_git_gateway.get_diff.assert_called_once_with(repo_path, staged=True)
        
        def should_fail_when_staging_fails(self, subject, mock_git_gateway, mock_llm_gateway, mock_generate_commit_message, repo_path):
            mock_git_gateway.add_all.return_value = (False, "Failed to stage")
            
            success, message, commit_msg = subject.commit_changes(repo_path, stage_all=True)
            
            assert success is False
            assert "Failed to stage changes: Failed to stage" in message
            assert commit_msg.short == ""
            assert commit_msg.long == ""
            mock_git_gateway.get_diff.assert_not_called()
            mock_generate_commit_message.assert_not_called()
            mock_git_gateway.commit.assert_not_called()
        
        def should_fail_when_diff_fails(self, subject, mock_git_gateway, mock_llm_gateway, mock_generate_commit_message, repo_path):
            mock_git_gateway.add_all.return_value = (True, "Successfully staged")
            mock_git_gateway.get_diff.return_value = ("", "fatal: not a git repository")
            
            success, message, commit_msg = subject.commit_changes(repo_path, stage_all=True)
            
            assert success is False
            assert "Failed to get diff: fatal: not a git repository" in message
            assert commit_msg.short == ""
            mock_generate_commit_message.assert_not_called()
            mock_git_gateway.commit.assert_not_called()
        
        def should_fail_when_no_staged_changes(self, subject, mock_git_gateway, mock_llm_gateway, mock_generate_commit_message, repo_path):
            mock_git_gateway.add_all.return_value = (True, "Successfully staged")
            mock_git_gateway.get_diff.return_value = ("", "")
            
            success, message, commit_msg = subject.commit_changes(repo_path, stage_all=True)
            
            assert success is False
            assert message == "No staged changes to commit"
            assert commit_msg.short == ""
            mock_generate_commit_message.assert_not_called()
            mock_git_gateway.commit.assert_not_called()
        
        def should_fail_when_llm_generation_fails(self, subject, mock_git_gateway, mock_llm_gateway, mock_generate_commit_message, repo_path):
            mock_git_gateway.add_all.return_value = (True, "Successfully staged")
            mock_git_gateway.get_diff.return_value = ("sample diff output", "")
            mock_generate_commit_message.side_effect = Exception("LLM error")
            
            success, message, commit_msg = subject.commit_changes(repo_path, stage_all=True)
            
            assert success is False
            assert "Failed to generate commit message: LLM error" in message
            assert commit_msg.short == ""
            mock_git_gateway.commit.assert_not_called()
        
        def should_fail_when_commit_fails(self, subject, mock_git_gateway, mock_llm_gateway, mock_generate_commit_message, repo_path, sample_commit_message):
            mock_git_gateway.add_all.return_value = (True, "Successfully staged")
            mock_git_gateway.get_diff.return_value = ("sample diff output", "")
            mock_generate_commit_message.return_value = sample_commit_message
            mock_git_gateway.commit.return_value = (False, "Nothing to commit")
            
            success, message, commit_msg = subject.commit_changes(repo_path, stage_all=True)
            
            assert success is False
            assert message == "Nothing to commit"
            assert commit_msg == sample_commit_message
    
    class DescribePreviewCommitMessage:
        
        def should_preview_commit_message_for_all_changes(self, subject, mock_git_gateway, mock_llm_gateway, mock_generate_commit_message, repo_path, sample_commit_message):
            mock_git_gateway.get_diff.return_value = ("sample diff output", "")
            mock_generate_commit_message.return_value = sample_commit_message
            
            success, message, commit_msg = subject.preview_commit_message(repo_path, stage_all=True)
            
            assert success is True
            assert message == "Commit message generated successfully"
            assert commit_msg == sample_commit_message
            mock_git_gateway.get_diff.assert_called_once_with(repo_path, staged=False)
            mock_generate_commit_message.assert_called_once_with(mock_llm_gateway, "sample diff output")
        
        def should_preview_commit_message_for_staged_changes(self, subject, mock_git_gateway, mock_llm_gateway, mock_generate_commit_message, repo_path, sample_commit_message):
            mock_git_gateway.get_diff.return_value = ("staged diff output", "")
            mock_generate_commit_message.return_value = sample_commit_message
            
            success, message, commit_msg = subject.preview_commit_message(repo_path, stage_all=False)
            
            assert success is True
            assert message == "Commit message generated successfully"
            assert commit_msg == sample_commit_message
            mock_git_gateway.get_diff.assert_called_once_with(repo_path, staged=True)
        
        def should_fail_when_no_changes_to_preview(self, subject, mock_git_gateway, mock_llm_gateway, mock_generate_commit_message, repo_path):
            mock_git_gateway.get_diff.return_value = ("", "")
            
            success, message, commit_msg = subject.preview_commit_message(repo_path, stage_all=True)
            
            assert success is False
            assert message == "No changes to preview"
            assert commit_msg.short == ""
            mock_generate_commit_message.assert_not_called()
        
        def should_fail_when_no_staged_changes_to_preview(self, subject, mock_git_gateway, mock_llm_gateway, mock_generate_commit_message, repo_path):
            mock_git_gateway.get_diff.return_value = ("", "")
            
            success, message, commit_msg = subject.preview_commit_message(repo_path, stage_all=False)
            
            assert success is False
            assert message == "No staged changes to preview"
            assert commit_msg.short == ""
            mock_generate_commit_message.assert_not_called()
        
        def should_fail_when_llm_generation_fails_during_preview(self, subject, mock_git_gateway, mock_llm_gateway, mock_generate_commit_message, repo_path):
            mock_git_gateway.get_diff.return_value = ("sample diff output", "")
            mock_generate_commit_message.side_effect = Exception("LLM error")
            
            success, message, commit_msg = subject.preview_commit_message(repo_path, stage_all=True)
            
            assert success is False
            assert "Failed to generate commit message: LLM error" in message
            assert commit_msg.short == ""
