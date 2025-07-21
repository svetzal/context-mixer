"""Tests for LLM gateway configuration functionality."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from context_mixer.config import Config
from context_mixer.gateways.llm_factory import LLMGatewayFactory


class DescribeConfig:
    """Test the Config class with LLM gateway settings."""

    def should_initialize_with_default_llm_settings(self):
        config = Config()
        
        assert config.llm_provider == "openai"
        assert config.llm_model == "o4-mini"
        assert config.llm_api_key is None

    def should_initialize_with_custom_llm_settings(self):
        config = Config(
            llm_provider="ollama",
            llm_model="phi3",
            llm_api_key="test-key"
        )
        
        assert config.llm_provider == "ollama"
        assert config.llm_model == "phi3"
        assert config.llm_api_key == "test-key"

    def should_convert_to_dict(self):
        config = Config(
            llm_provider="ollama",
            llm_model="phi3",
            llm_api_key="test-key"
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["llm_provider"] == "ollama"
        assert config_dict["llm_model"] == "phi3"
        assert config_dict["llm_api_key"] == "test-key"

    def should_create_from_dict(self):
        config_dict = {
            "library_path": "/tmp/test",
            "llm_provider": "ollama",
            "llm_model": "phi3",
            "llm_api_key": "test-key",
            "clustering_enabled": False
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.llm_provider == "ollama"
        assert config.llm_model == "phi3"
        assert config.llm_api_key == "test-key"
        assert config.clustering_enabled is False

    def should_save_and_load_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        
        # Create config with custom settings
        original_config = Config(
            library_path=tmp_path,
            llm_provider="ollama",
            llm_model="phi3",
            llm_api_key="test-key"
        )
        
        # Save config
        original_config.save()
        
        # Load config
        loaded_config = Config.load(config_path)
        
        assert loaded_config.llm_provider == "ollama"
        assert loaded_config.llm_model == "phi3"
        assert loaded_config.llm_api_key == "test-key"

    def should_handle_corrupted_config_file(self, tmp_path):
        config_path = tmp_path / "config.json"
        
        # Create corrupted config file
        with open(config_path, 'w') as f:
            f.write("invalid json")
        
        # Should return default config when file is corrupted
        config = Config.load(config_path)
        
        assert config.llm_provider == "openai"
        assert config.llm_model == "o4-mini"

    def should_handle_missing_config_file(self, tmp_path):
        config_path = tmp_path / "nonexistent.json"
        
        # Should return default config when file doesn't exist
        config = Config.load(config_path)
        
        assert config.llm_provider == "openai"
        assert config.llm_model == "o4-mini"


class DescribeLLMGatewayFactory:
    """Test the LLM gateway factory functions."""

    @patch('context_mixer.gateways.llm_factory.OpenAIGateway')
    @patch('context_mixer.gateways.llm_factory.LLMGateway')
    def should_create_openai_gateway_with_api_key(self, mock_llm_gateway, mock_openai_gateway):
        config = Config(
            llm_provider="openai",
            llm_model="gpt-4",
            llm_api_key="test-api-key"
        )
        
        mock_openai_instance = MagicMock()
        mock_openai_gateway.return_value = mock_openai_instance
        
        LLMGatewayFactory.create_gateway(config)
        
        mock_openai_gateway.assert_called_once_with(api_key="test-api-key")
        mock_llm_gateway.assert_called_once_with(model="gpt-4", gateway=mock_openai_instance)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'env-api-key'})
    @patch('context_mixer.gateways.llm_factory.OpenAIGateway')
    @patch('context_mixer.gateways.llm_factory.LLMGateway')
    def should_use_env_api_key_for_openai_when_not_in_config(self, mock_llm_gateway, mock_openai_gateway):
        config = Config(
            llm_provider="openai",
            llm_model="gpt-4",
            llm_api_key=None
        )
        
        mock_openai_instance = MagicMock()
        mock_openai_gateway.return_value = mock_openai_instance
        
        LLMGatewayFactory.create_gateway(config)
        
        mock_openai_gateway.assert_called_once_with(api_key="env-api-key")

    def should_raise_error_for_openai_without_api_key(self):
        config = Config(
            llm_provider="openai",
            llm_model="gpt-4",
            llm_api_key=None
        )
        
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI provider requires an API key"):
                LLMGatewayFactory.create_gateway(config)

    @patch('context_mixer.gateways.llm_factory.OllamaGateway')
    @patch('context_mixer.gateways.llm_factory.LLMGateway')
    def should_create_ollama_gateway_without_api_key(self, mock_llm_gateway, mock_ollama_gateway):
        config = Config(
            llm_provider="ollama",
            llm_model="phi3",
            llm_api_key=None
        )
        
        mock_ollama_instance = MagicMock()
        mock_ollama_gateway.return_value = mock_ollama_instance
        
        LLMGatewayFactory.create_gateway(config)
        
        mock_ollama_gateway.assert_called_once_with()
        mock_llm_gateway.assert_called_once_with(model="phi3", gateway=mock_ollama_instance)

    def should_raise_error_for_unsupported_provider(self):
        config = Config(
            llm_provider="unsupported",
            llm_model="some-model",
            llm_api_key=None
        )
        
        with pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"):
            LLMGatewayFactory.create_gateway(config)

    @patch('context_mixer.gateways.llm_factory.Config')
    @patch('context_mixer.gateways.llm_factory.LLMGatewayFactory.create_gateway')
    def should_create_default_gateway_from_saved_config(self, mock_create_gateway, mock_config_class):
        mock_config_instance = MagicMock()
        mock_config_class.load.return_value = mock_config_instance
        
        LLMGatewayFactory.create_default_gateway()
        
        mock_config_class.load.assert_called_once()
        mock_create_gateway.assert_called_once_with(mock_config_instance)