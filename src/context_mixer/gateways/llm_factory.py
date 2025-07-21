"""
Factory for creating LLM gateways based on configuration.

This module provides a factory function to create the appropriate LLM gateway
based on the configuration settings.
"""

import os
from typing import Optional

from mojentic.llm.gateways import OpenAIGateway, OllamaGateway

from .llm import LLMGateway
from ..config import Config


class LLMGatewayFactory:
    """
    Factory for creating LLM gateways based on configuration.
    
    This factory follows the same pattern as other factories in the codebase
    to provide consistent architecture.
    """
    
    @staticmethod
    def create_gateway(config: Config) -> LLMGateway:
        """
        Create an LLM gateway based on the configuration.

        Args:
            config: Configuration object containing LLM settings.

        Returns:
            LLMGateway: Configured LLM gateway.

        Raises:
            ValueError: If an unsupported provider is specified or required API key is missing.
        """
        provider = config.llm_provider.lower()
        model = config.llm_model
        api_key = config.llm_api_key

        if provider == "openai":
            # For OpenAI, we need an API key
            if api_key is None:
                # Try to get from environment as fallback
                api_key = os.environ.get("OPENAI_API_KEY")
            
            if api_key is None:
                raise ValueError(
                    "OpenAI provider requires an API key. Please set it in the configuration "
                    "or provide it via the OPENAI_API_KEY environment variable."
                )
            
            openai_gateway = OpenAIGateway(api_key=api_key)
            return LLMGateway(model=model, gateway=openai_gateway)
        
        elif provider == "ollama":
            # Ollama doesn't require an API key
            ollama_gateway = OllamaGateway()
            return LLMGateway(model=model, gateway=ollama_gateway)
        
        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers are: openai, ollama"
            )

    @classmethod
    def create_default_gateway(cls) -> LLMGateway:
        """
        Create a default LLM gateway using the saved configuration.

        Returns:
            LLMGateway: Default configured LLM gateway.
        """
        config = Config.load()
        return cls.create_gateway(config)