"""
Tests for commit message generation operations.
"""

import pytest
from mojentic.llm import MessageRole

from prompt_mixer.commands.operations.commit_message import generate_commit_message
from prompt_mixer.gateways.llm import LLMGateway
from prompt_mixer.domain.commit_message import CommitMessage


@pytest.fixture
def mock_llm_gateway(mocker):
    return mocker.MagicMock(spec=LLMGateway)


class DescribeGenerateCommitMessage:
    
    def should_generate_commit_message_from_diff(self, mock_llm_gateway):
        sample_diff = """diff --git a/src/feature.py b/src/feature.py
index 1234567..abcdefg 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,6 @@
 def existing_function():
     return "existing"
+
+def new_function():
+    return "new feature"
"""
        expected_commit_message = CommitMessage(
            short="Add new function to feature module",
            long="Implemented new_function() that provides new feature functionality to complement the existing function."
        )
        mock_llm_gateway.generate_object.return_value = expected_commit_message
        
        result = generate_commit_message(mock_llm_gateway, sample_diff)
        
        assert result == expected_commit_message
        mock_llm_gateway.generate_object.assert_called_once()
        call_args = mock_llm_gateway.generate_object.call_args
        messages, object_model = call_args[0]
        
        assert object_model == CommitMessage
        assert len(messages) == 2
        assert messages[0].role == MessageRole.System
        assert "commit messages from diffs" in messages[0].content
        assert messages[1].role == MessageRole.User
        assert sample_diff in messages[1].content
    
    def should_handle_empty_diff(self, mock_llm_gateway):
        empty_diff = ""
        expected_commit_message = CommitMessage(
            short="No changes detected",
            long="The diff appears to be empty with no changes to commit."
        )
        mock_llm_gateway.generate_object.return_value = expected_commit_message
        
        result = generate_commit_message(mock_llm_gateway, empty_diff)
        
        assert result == expected_commit_message
        mock_llm_gateway.generate_object.assert_called_once()
    
    def should_include_proper_system_prompt(self, mock_llm_gateway):
        sample_diff = "diff --git a/test.txt b/test.txt"
        mock_llm_gateway.generate_object.return_value = CommitMessage(
            short="Test commit", 
            long="Test commit message"
        )
        
        generate_commit_message(mock_llm_gateway, sample_diff)
        
        call_args = mock_llm_gateway.generate_object.call_args
        messages = call_args[0][0]
        system_message = messages[0]
        
        assert system_message.role == MessageRole.System
        assert "50 characters or less" in system_message.content
        assert "imperative mood" in system_message.content
        assert "present tense" in system_message.content
    
    def should_propagate_llm_exceptions(self, mock_llm_gateway):
        sample_diff = "diff --git a/test.txt b/test.txt"
        mock_llm_gateway.generate_object.side_effect = Exception("LLM error")
        
        with pytest.raises(Exception, match="LLM error"):
            generate_commit_message(mock_llm_gateway, sample_diff)
