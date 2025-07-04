from pathlib import Path
from rich.panel import Panel


def do_init(console, config, remote=None, provider=None, model=None, git_gateway=None):
    """
    Initialize a new prompt library.

    Args:
        console: Rich console for output
        config: Config object containing the library path
        remote: URL of remote Git repository to link
        provider: LLM provider
        model: LLM model
        git_gateway: GitGateway instance for git operations
    """
    path = config.library_path

    console.print(
        Panel(f"Initializing prompt library at [bold]{path}[/bold]", title="Context Mixer"))

    # Create the directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)

    if remote:
        # Clone the remote repository
        console.print(f"Cloning repository from [bold]{remote}[/bold]...")
        _, message = git_gateway.clone(remote, path)
        console.print(message)
    else:
        # Initialize a new Git repository
        console.print("Initializing new Git repository...")
        _, message = git_gateway.init(path)
        console.print(message)

    # TODO: Set up provider and model configuration
    if provider or model:
        console.print("Provider and model configuration is not yet implemented.")
