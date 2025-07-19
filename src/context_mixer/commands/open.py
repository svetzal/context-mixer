"""
Implementation of the open command for Context Mixer.

This module provides the functionality to open the prompt library in the default editor.
"""

import os
import subprocess
from pathlib import Path

from rich.panel import Panel

from .base import Command, CommandContext, CommandResult


class OpenCommand(Command):
    """
    Command for opening the prompt library in the default editor.

    Implements the Command pattern as specified in the architectural improvements backlog.
    """

    async def execute(self, context: CommandContext) -> CommandResult:
        """
        Execute the open command with the given context.

        Args:
            context: CommandContext containing console and config

        Returns:
            CommandResult indicating success/failure
        """
        try:
            # Call the existing implementation for backward compatibility
            do_open(
                console=context.console,
                config=context.config
            )

            return CommandResult(
                success=True,
                message="Library opened successfully"
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to open library: {str(e)}",
                error=e
            )


def do_open(console, config):
    """
    Open the prompt library in the default editor.

    Args:
        console: Rich console for output
        config: Config object containing the library path
    """
    editor = os.environ.get("EDITOR", "code")
    path = config.library_path

    console.print(
        Panel(f"Opening prompt library with: [bold]{editor}[/bold]", title="Context Mixer"))

    try:
        # Run the editor command
        subprocess.run([editor, str(path)], check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error:[/bold red] Failed to open editor: {e}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
