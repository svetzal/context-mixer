# LLM Gateway Configuration

Context Mixer now supports both OpenAI and Ollama LLM providers, giving users the flexibility to choose their preferred language model backend.

## Quick Start

### For OpenAI Users

```bash
# Configure OpenAI with your API key
cmx config --provider openai --model gpt-4 --api-key YOUR_API_KEY

# Or use a smaller/faster model
cmx config --provider openai --model o4-mini --api-key YOUR_API_KEY
```

### For Ollama Users

```bash
# Configure Ollama (no API key required)
cmx config --provider ollama --model phi3

# Or use other Ollama models
cmx config --provider ollama --model llama2
cmx config --provider ollama --model codellama
```

## Configuration Management

### View Current Configuration

```bash
cmx config --show
```

This displays:
- Library path
- LLM provider
- LLM model
- API key status (without revealing the key)
- Clustering settings
- Batch size

### Update Configuration

You can update individual settings without affecting others:

```bash
# Change just the model
cmx config --model gpt-4

# Change just the provider
cmx config --provider ollama

# Change multiple settings at once
cmx config --provider openai --model o4-mini --api-key new-key
```

## Configuration Storage

Configuration is stored in `~/.context-mixer/config.json` and persists across CLI sessions. The configuration file contains:

```json
{
  "library_path": "/home/user/.context-mixer",
  "conflict_detection_batch_size": 5,
  "clustering_enabled": true,
  "min_cluster_size": 3,
  "clustering_fallback": true,
  "llm_provider": "ollama",
  "llm_model": "phi3",
  "llm_api_key": null
}
```

## API Key Handling

For OpenAI, the system checks for API keys in this order:
1. Configuration file (`cmx config --api-key YOUR_KEY`)
2. Environment variable (`OPENAI_API_KEY`)
3. Error if neither is found

## Error Handling

If no valid LLM configuration is found when running commands that require LLM functionality, you'll see helpful guidance:

```
Error: OpenAI provider requires an API key. Please set it in the configuration 
or provide it via the OPENAI_API_KEY environment variable.

To configure the LLM gateway, run:
  cmx config --provider openai --model o4-mini --api-key YOUR_API_KEY
  cmx config --provider ollama --model phi3

To see current configuration:
  cmx config --show
```

## Supported Providers

- **OpenAI**: Requires API key, supports all OpenAI models (gpt-4, o4-mini, etc.)
- **Ollama**: No API key required, supports all locally available Ollama models

## Migration from Previous Versions

Existing installations will continue to work with OpenAI as the default provider. The system will attempt to use the `OPENAI_API_KEY` environment variable if no configuration file exists.

To explicitly configure your setup:

```bash
# For existing OpenAI users
cmx config --provider openai --model o4-mini --api-key $OPENAI_API_KEY

# For users wanting to switch to Ollama
cmx config --provider ollama --model phi3
```