"""
Implementation of the open command for Context Mixer.

This module provides the functionality to open the prompt library in the default editor.
"""

import os
import subprocess
from pathlib import Path

from rich.panel import Panel


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