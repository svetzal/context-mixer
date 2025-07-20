from rich.panel import Panel

from .base import Command, CommandContext, CommandResult


class InitCommand(Command):
    """
    Command for initializing a new prompt library.

    Implements the Command pattern as specified in the architectural improvements backlog.
    """

    async def execute(self, context: CommandContext) -> CommandResult:
        """
        Execute the init command with the given context.

        Args:
            context: CommandContext containing console, config, and parameters

        Returns:
            CommandResult indicating success/failure
        """
        try:
            # Extract parameters from context
            remote = context.parameters.get('remote')
            provider = context.parameters.get('provider')
            model = context.parameters.get('model')

            # Call the existing implementation for backward compatibility
            do_init(
                console=context.console,
                config=context.config,
                remote=remote,
                provider=provider,
                model=model,
                git_gateway=context.git_gateway
            )

            return CommandResult(
                success=True,
                message="Library initialized successfully"
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to initialize library: {str(e)}",
                error=e
            )


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
