from pathlib import Path
from rich.panel import Panel


def do_init(console, path, remote, provider, model, git_gateway):
    """
    Initialize a new prompt library.

    Args:
        console: Rich console for output
        path: Path where the prompt library should be initialized
        remote: URL of remote Git repository to link
        provider: LLM provider
        model: LLM model
        git_gateway: GitGateway instance for git operations
    """
    console.print(
        Panel(f"Initializing prompt library at [bold]{path}[/bold]", title="Prompt Mixer"))

    # Create the directory if it doesn't exist
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if remote:
        # Clone the remote repository
        console.print(f"Cloning repository from [bold]{remote}[/bold]...")
        success, message = git_gateway.clone(remote, path)
        console.print(message)
    else:
        # Initialize a new Git repository
        console.print("Initializing new Git repository...")
        success, message = git_gateway.init(path)
        console.print(message)

    # TODO: Set up provider and model configuration
    if provider or model:
        console.print("Provider and model configuration is not yet implemented.")
