import pytest
from typing import List
from datetime import datetime
from pydantic import BaseModel

from mojentic.llm import LLMMessage, MessageRole
from context_mixer.gateways.llm import LLMGateway, MockLLMGateway, LLMCall


class SampleModel(BaseModel):
    """Sample Pydantic model for testing generate_object."""
    name: str = "test"
    value: int = 42


@pytest.fixture
def sample_messages():
    """Sample LLM messages for testing."""
    return [
        LLMMessage(role=MessageRole.System, content="You are a helpful assistant"),
        LLMMessage(role=MessageRole.User, content="Hello, world!")
    ]


@pytest.fixture
def mock_gateway():
    """Basic mock gateway with no configured responses."""
    return MockLLMGateway()


@pytest.fixture
def configured_mock_gateway():
    """Mock gateway with configured responses."""
    responses = {
        'generate': "Custom mock response",
        'generate_embeddings': [0.9, 0.8, 0.7, 0.6, 0.5],
        'generate_object': SampleModel(name="configured", value=100)
    }
    return MockLLMGateway(responses=responses)


class DescribeLLMCall:
    def should_create_call_with_timestamp(self):
        call = LLMCall(
            method="test_method",
            args=("arg1", "arg2"),
            kwargs={"key": "value"}
        )

        assert call.method == "test_method"
        assert call.args == ("arg1", "arg2")
        assert call.kwargs == {"key": "value"}
        assert isinstance(call.timestamp, datetime)
        assert call.response is None

    def should_preserve_provided_timestamp(self):
        test_time = datetime(2023, 1, 1, 12, 0, 0)
        call = LLMCall(
            method="test_method",
            args=(),
            kwargs={},
            timestamp=test_time
        )

        assert call.timestamp == test_time


class DescribeMockLLMGateway:
    def should_initialize_with_empty_responses(self, mock_gateway):
        assert mock_gateway.responses == {}
        assert mock_gateway.call_log == []

    def should_initialize_with_configured_responses(self, configured_mock_gateway):
        assert "generate" in configured_mock_gateway.responses
        assert configured_mock_gateway.responses["generate"] == "Custom mock response"

    def should_log_generate_calls(self, mock_gateway, sample_messages):
        result = mock_gateway.generate(sample_messages)

        assert result == "Mock LLM response"  # Default response
        assert mock_gateway.get_call_count() == 1
        assert mock_gateway.get_call_count("generate") == 1

        calls = mock_gateway.get_calls("generate")
        assert len(calls) == 1
        assert calls[0].method == "generate"
        assert calls[0].args == (sample_messages,)
        assert calls[0].response == "Mock LLM response"

    def should_use_configured_generate_response(self, configured_mock_gateway, sample_messages):
        result = configured_mock_gateway.generate(sample_messages)

        assert result == "Custom mock response"
        assert configured_mock_gateway.get_call_count("generate") == 1

    def should_log_generate_object_calls(self, mock_gateway, sample_messages):
        result = mock_gateway.generate_object(sample_messages, SampleModel)

        assert isinstance(result, SampleModel)
        assert result.name == "test"  # Default value
        assert result.value == 42     # Default value

        assert mock_gateway.get_call_count("generate_object") == 1
        calls = mock_gateway.get_calls("generate_object")
        assert calls[0].method == "generate_object"
        assert calls[0].args == (sample_messages, SampleModel)

    def should_use_configured_generate_object_response(self, configured_mock_gateway, sample_messages):
        result = configured_mock_gateway.generate_object(sample_messages, SampleModel)

        assert isinstance(result, SampleModel)
        assert result.name == "configured"
        assert result.value == 100

    def should_log_generate_embeddings_calls(self, mock_gateway):
        content = "Test content for embeddings"
        result = mock_gateway.generate_embeddings(content)

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]  # Default response
        assert mock_gateway.get_call_count("generate_embeddings") == 1

        calls = mock_gateway.get_calls("generate_embeddings")
        assert calls[0].method == "generate_embeddings"
        assert calls[0].args == (content,)

    def should_use_configured_embeddings_response(self, configured_mock_gateway):
        content = "Test content"
        result = configured_mock_gateway.generate_embeddings(content)

        assert result == [0.9, 0.8, 0.7, 0.6, 0.5]

    def should_support_sequence_of_responses(self):
        responses = {
            'generate': ["First response", "Second response", "Third response"]
        }
        mock_gateway = MockLLMGateway(responses=responses)
        messages = [LLMMessage(role=MessageRole.User, content="Test")]

        assert mock_gateway.generate(messages) == "First response"
        assert mock_gateway.generate(messages) == "Second response"
        assert mock_gateway.generate(messages) == "Third response"

        # After exhausting the list, should return default
        assert mock_gateway.generate(messages) == "Mock LLM response"

    def should_track_multiple_method_calls(self, mock_gateway, sample_messages):
        mock_gateway.generate(sample_messages)
        mock_gateway.generate_embeddings("test content")
        mock_gateway.generate_object(sample_messages, SampleModel)

        assert mock_gateway.get_call_count() == 3
        assert mock_gateway.get_call_count("generate") == 1
        assert mock_gateway.get_call_count("generate_embeddings") == 1
        assert mock_gateway.get_call_count("generate_object") == 1

    def should_check_if_method_was_called(self, mock_gateway, sample_messages):
        assert not mock_gateway.was_called("generate")

        mock_gateway.generate(sample_messages)

        assert mock_gateway.was_called("generate")
        assert not mock_gateway.was_called("generate_embeddings")

    def should_verify_method_called_with_arguments(self, mock_gateway, sample_messages):
        mock_gateway.generate(sample_messages)

        assert mock_gateway.assert_called_with("generate", sample_messages)
        assert not mock_gateway.assert_called_with("generate", [])

    def should_clear_call_log(self, mock_gateway, sample_messages):
        mock_gateway.generate(sample_messages)
        assert mock_gateway.get_call_count() == 1

        mock_gateway.clear_calls()
        assert mock_gateway.get_call_count() == 0
        assert not mock_gateway.was_called("generate")

    def should_filter_calls_by_method(self, mock_gateway, sample_messages):
        mock_gateway.generate(sample_messages)
        mock_gateway.generate_embeddings("test")
        mock_gateway.generate(sample_messages)

        all_calls = mock_gateway.get_calls()
        generate_calls = mock_gateway.get_calls("generate")
        embedding_calls = mock_gateway.get_calls("generate_embeddings")

        assert len(all_calls) == 3
        assert len(generate_calls) == 2
        assert len(embedding_calls) == 1

        assert all(call.method == "generate" for call in generate_calls)
        assert all(call.method == "generate_embeddings" for call in embedding_calls)

    def should_handle_default_response_fallback(self):
        responses = {'default': 'Default fallback response'}
        mock_gateway = MockLLMGateway(responses=responses)
        messages = [LLMMessage(role=MessageRole.User, content="Test")]

        # generate should use its built-in default, not the fallback
        assert mock_gateway.generate(messages) == "Mock LLM response"

        # But if we had a custom method, it would use the fallback
        # (This is more of a design verification than a real test case)

    def should_raise_error_for_unconfigured_generate_object(self):
        class ComplexModel(BaseModel):
            required_field: str  # No default value

        mock_gateway = MockLLMGateway()
        messages = [LLMMessage(role=MessageRole.User, content="Test")]

        with pytest.raises(ValueError, match="No response configured for generate_object"):
            mock_gateway.generate_object(messages, ComplexModel)

    def should_provide_same_interface_as_llm_gateway(self):
        mock_gateway = MockLLMGateway()
        # Verify MockLLMGateway provides the same interface as LLMGateway
        # without inheritance coupling
        assert hasattr(mock_gateway, 'generate')
        assert hasattr(mock_gateway, 'generate_object')
        assert hasattr(mock_gateway, 'generate_embeddings')
        assert callable(mock_gateway.generate)
        assert callable(mock_gateway.generate_object)
        assert callable(mock_gateway.generate_embeddings)

    def should_support_scenario_responses(self):
        responses = {
            'generate_scenario': ["Scenario 1", "Scenario 2"]
        }
        mock_gateway = MockLLMGateway(responses=responses)
        messages = [LLMMessage(role=MessageRole.User, content="Test")]

        assert mock_gateway.generate(messages) == "Scenario 1"
        assert mock_gateway.generate(messages) == "Scenario 2"
        # After scenarios are exhausted, fall back to default
        assert mock_gateway.generate(messages) == "Mock LLM response"
