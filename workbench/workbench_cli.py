#!/usr/bin/env python3
"""
Conflict Detection and Resolution Workbench

This script runs automated integration tests for conflict detection and resolution scenarios.
It provides a persistent testing environment that accumulates troublesome scenarios
and ensures we can always pass them all.
"""

import argparse
import asyncio
import importlib
import importlib.util
import inspect
import logging
import os
import pkgutil
import sys
import tempfile
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logging.basicConfig(level=logging.WARN)

# Add the src directory and project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from context_mixer.config import Config
from context_mixer.gateways.llm import LLMGateway, OpenAIModels
from context_mixer.commands.ingest import do_ingest
from workbench.automated_resolver import AutomatedConflictResolver
from mojentic.llm.gateways import OpenAIGateway


def discover_scenarios() -> Dict[str, Any]:
    """Dynamically discover all scenario modules."""
    scenarios = {}

    # Find the scenarios package directory
    scenarios_dir = Path(__file__).parent / "scenarios"
    if not scenarios_dir.exists():
        return scenarios

    # Import each scenario module and look for get_scenario function
    for file_path in scenarios_dir.glob("*.py"):
        if file_path.name.startswith("__"):
            continue

        module_name = file_path.stem
        try:
            # Import the module dynamically
            spec = importlib.util.spec_from_file_location(f"scenarios.{module_name}", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for get_scenario function
                if hasattr(module, 'get_scenario'):
                    get_scenario_func = getattr(module, 'get_scenario')
                    if callable(get_scenario_func):
                        scenarios[module_name] = get_scenario_func

        except Exception:
            # Silently skip modules that can't be loaded
            continue

    return scenarios


class WorkbenchRunner:
    """Main workbench runner that executes conflict detection scenarios."""

    def __init__(self):
        """Initialize the workbench runner."""
        self.console = Console()
        self.automated_resolver = AutomatedConflictResolver(self.console)

        # Configure model selection via environment variable
        model_name = os.environ.get("WORKBENCH_MODEL", "O4_MINI").upper()
        try:
            self.model = OpenAIModels[model_name]
        except KeyError:
            self.console.print(f"[red]Error: Unknown model '{model_name}'. Available models: {', '.join(OpenAIModels.__members__.keys())}[/red]")
            sys.exit(1)

        # Configure OpenAI gateway
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
            sys.exit(1)

        openai_gateway = OpenAIGateway(api_key=api_key)
        self.llm_gateway = LLMGateway(model=self.model.value, gateway=openai_gateway)

        # Auto-discover scenarios
        self.scenarios = discover_scenarios()
        if not self.scenarios:
            self.console.print("[red]Error: No scenarios found in scenarios directory[/red]")

    async def run_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Run a single scenario.

        Args:
            scenario_name: Name of the scenario to run

        Returns:
            Dictionary with scenario results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        # Start timing the scenario
        start_time = time.time()

        scenario = self.scenarios[scenario_name]()
        self.console.print(f"\n[bold blue]Running scenario: {scenario.name}[/bold blue]")
        self.console.print(f"Description: {scenario.description}")

        # Create temporary directory for this scenario
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            library_path = temp_path / "library"

            # Create test files
            for filename, content in scenario.input_files.items():
                file_path = temp_path / filename
                file_path.write_text(content)

            # Setup config
            config = Config(library_path=library_path)

            # Track results
            results = {
                "scenario_name": scenario_name,
                "description": scenario.description,
                "passed": False,
                "conflicts_detected": 0,
                "conflicts_expected": len(scenario.expected_conflicts),
                "validation_results": [],
                "chunk_count_results": [],
                "errors": [],
                "execution_time_seconds": 0.0
            }

            try:
                # Check if chunk count validation is needed
                track_chunks = scenario.expected_chunk_counts is not None

                if track_chunks:
                    # Process all files together with chunk tracking
                    self.console.print("[cyan]Ingesting files with chunk count tracking...[/cyan]")
                    chunk_counts = await do_ingest(
                        console=self.console,
                        config=config,
                        llm_gateway=self.llm_gateway,
                        path=temp_path,
                        project_id="workbench-test",
                        project_name="Workbench Test",
                        commit=False,
                        detect_boundaries=True,
                        resolver=self.automated_resolver,
                        track_chunks_per_file=True
                    )

                    # Validate chunk counts
                    if chunk_counts is not None:
                        results["chunk_count_results"] = self._validate_chunk_counts(
                            chunk_counts, scenario.expected_chunk_counts
                        )
                    else:
                        results["errors"].append("Failed to get chunk counts from ingestion")
                else:
                    # Original behavior: ingest files one by one
                    for filename in scenario.input_files.keys():
                        file_path = temp_path / filename

                        self.console.print(f"[cyan]Ingesting {filename}...[/cyan]")
                        await do_ingest(
                            console=self.console,
                            config=config,
                            llm_gateway=self.llm_gateway,
                            path=file_path,
                            project_id="workbench-test",
                            project_name="Workbench Test",
                            commit=False,
                            detect_boundaries=True,
                            resolver=self.automated_resolver
                        )

                # Check if context.md was created and validate content
                context_file = library_path / "context.md"
                if context_file.exists():
                    content = context_file.read_text()
                    results["validation_results"] = self._validate_content(content, scenario.validation_checks)

                    # Calculate overall pass status including chunk count validation
                    content_validation_passed = all(result["passed"] for result in results["validation_results"])
                    chunk_count_validation_passed = all(result["passed"] for result in results["chunk_count_results"]) if results["chunk_count_results"] else True
                    results["passed"] = content_validation_passed and chunk_count_validation_passed
                else:
                    results["errors"].append("context.md was not created")

            except Exception as e:
                results["errors"].append(str(e))
                self.console.print(f"[red]Error running scenario: {e}[/red]")

        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        results["execution_time_seconds"] = execution_time

        # Display timing information
        self.console.print(f"[dim]⏱️  Scenario completed in {execution_time:.2f} seconds[/dim]")

        return results

    def _validate_content(self, content: str, validation_checks: List[str]) -> List[Dict[str, Any]]:
        """
        Validate content against validation checks.

        Args:
            content: Content to validate
            validation_checks: List of validation check strings

        Returns:
            List of validation results
        """
        results = []

        for check in validation_checks:
            if check.startswith("should_contain:"):
                expected_text = check[15:]  # Remove "should_contain:" prefix
                passed = expected_text in content
                results.append({
                    "check": check,
                    "passed": passed,
                    "message": f"Content {'contains' if passed else 'missing'}: '{expected_text}'"
                })
            elif check.startswith("should_not_contain:"):
                unexpected_text = check[19:]  # Remove "should_not_contain:" prefix
                passed = unexpected_text not in content
                results.append({
                    "check": check,
                    "passed": passed,
                    "message": f"Content {'correctly excludes' if passed else 'incorrectly contains'}: '{unexpected_text}'"
                })

        return results

    def _validate_chunk_counts(self, actual_counts: Dict[str, int], expected_counts: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Validate actual chunk counts against expected counts.

        Args:
            actual_counts: Dictionary mapping filename to actual chunk count
            expected_counts: Dictionary mapping filename to expected chunk count

        Returns:
            List of chunk count validation results
        """
        results = []

        for filename, expected_count in expected_counts.items():
            actual_count = actual_counts.get(filename, 0)
            passed = actual_count == expected_count

            results.append({
                "filename": filename,
                "expected_count": expected_count,
                "actual_count": actual_count,
                "passed": passed,
                "message": f"File '{filename}': expected {expected_count} chunks, got {actual_count}"
            })

        return results

    async def run_all_scenarios(self) -> Dict[str, Any]:
        """
        Run all available scenarios.

        Returns:
            Dictionary with overall results
        """
        # Start timing the overall execution
        overall_start_time = time.time()

        self.console.print("[bold green]🚀 Starting Conflict Detection Workbench[/bold green]")
        self.console.print(f"Using OpenAI model: {self.model}")

        overall_results = {
            "total_scenarios": len(self.scenarios),
            "passed_scenarios": 0,
            "failed_scenarios": 0,
            "scenario_results": [],
            "total_execution_time_seconds": 0.0
        }

        for scenario_name in self.scenarios.keys():
            try:
                result = await self.run_scenario(scenario_name)
                overall_results["scenario_results"].append(result)

                if result["passed"]:
                    overall_results["passed_scenarios"] += 1
                    self.console.print(f"[green]✅ {scenario_name} PASSED[/green]")
                else:
                    overall_results["failed_scenarios"] += 1
                    self.console.print(f"[red]❌ {scenario_name} FAILED[/red]")

                    # Show validation failures
                    for validation in result["validation_results"]:
                        if not validation["passed"]:
                            self.console.print(f"  [red]• {validation['message']}[/red]")

                    # Show chunk count validation failures
                    for chunk_validation in result.get("chunk_count_results", []):
                        if not chunk_validation["passed"]:
                            self.console.print(f"  [red]• {chunk_validation['message']}[/red]")

                    # Show errors
                    for error in result["errors"]:
                        self.console.print(f"  [red]• Error: {error}[/red]")

            except Exception as e:
                overall_results["failed_scenarios"] += 1
                self.console.print(f"[red]❌ {scenario_name} FAILED with exception: {e}[/red]")

        # Calculate total execution time
        overall_end_time = time.time()
        total_execution_time = overall_end_time - overall_start_time
        overall_results["total_execution_time_seconds"] = total_execution_time

        # Display overall timing information
        self.console.print(f"\n[bold dim]⏱️  Total execution time: {total_execution_time:.2f} seconds[/bold dim]")

        return overall_results

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the workbench results."""

        # Create summary table
        table = Table(title="Workbench Results Summary")
        table.add_column("Scenario", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Content Validations", style="dim")
        table.add_column("Chunk Counts", style="dim")
        table.add_column("Time (s)", style="yellow", justify="right")

        for result in results["scenario_results"]:
            status = "[green]PASSED[/green]" if result["passed"] else "[red]FAILED[/red]"

            # Content validation summary
            validation_count = len(result["validation_results"])
            passed_validations = sum(1 for v in result["validation_results"] if v["passed"])
            validations = f"{passed_validations}/{validation_count}"

            # Chunk count validation summary
            chunk_count_results = result.get("chunk_count_results", [])
            if chunk_count_results:
                chunk_count_passed = sum(1 for c in chunk_count_results if c["passed"])
                chunk_counts = f"{chunk_count_passed}/{len(chunk_count_results)}"
            else:
                chunk_counts = "N/A"

            execution_time = f"{result.get('execution_time_seconds', 0.0):.2f}"

            table.add_row(result["scenario_name"], status, validations, chunk_counts, execution_time)

        self.console.print(table)

        # Overall summary
        total = results["total_scenarios"]
        passed = results["passed_scenarios"]
        failed = results["failed_scenarios"]

        if passed == total:
            panel_style = "green"
            status_text = "🎉 ALL SCENARIOS PASSED!"
        else:
            panel_style = "red"
            status_text = f"⚠️  {failed} of {total} scenarios failed"

        total_time = results.get("total_execution_time_seconds", 0.0)
        summary_text = f"""
{status_text}

Passed: {passed}/{total}
Failed: {failed}/{total}
Total Time: {total_time:.2f} seconds
        """

        self.console.print(Panel(summary_text.strip(), title="Final Results", style=panel_style))


async def run_command(args):
    """Run scenarios command."""
    runner = WorkbenchRunner()

    if args.scenario:
        # Run specific scenario
        result = await runner.run_scenario(args.scenario)
        if result["passed"]:
            print(f"✅ Scenario {args.scenario} PASSED")
            sys.exit(0)
        else:
            print(f"❌ Scenario {args.scenario} FAILED")
            sys.exit(1)
    else:
        # Run all scenarios
        results = await runner.run_all_scenarios()
        runner.print_summary(results)

        if results["failed_scenarios"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)


def list_scenarios_command(args):
    """List all available scenarios."""
    console = Console()
    runner = WorkbenchRunner()

    console.print("[bold green]Available Scenarios[/bold green]")

    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="dim")

    for scenario_name in runner.scenarios.keys():
        scenario = runner.scenarios[scenario_name]()
        table.add_row(scenario_name, scenario.description)

    console.print(table)


def validate_scenario_command(args):
    """Validate a specific scenario definition."""
    console = Console()
    runner = WorkbenchRunner()

    if args.scenario_name not in runner.scenarios:
        console.print(f"[red]Error: Scenario '{args.scenario_name}' not found[/red]")
        sys.exit(1)

    try:
        scenario = runner.scenarios[args.scenario_name]()
        console.print(f"[green]✅ Scenario '{args.scenario_name}' is valid[/green]")
        console.print(f"Description: {scenario.description}")
        console.print(f"Input files: {len(scenario.input_files)}")
        console.print(f"Expected conflicts: {len(scenario.expected_conflicts)}")
        console.print(f"Validation checks: {len(scenario.validation_checks)}")
    except Exception as e:
        console.print(f"[red]❌ Scenario '{args.scenario_name}' validation failed: {e}[/red]")
        sys.exit(1)


def load_yaml_scenario(yaml_path: Path):
    """Load a scenario from YAML file and create a Python scenario file."""
    try:
        with yaml_path.open('r') as f:
            yaml_data = yaml.safe_load(f)

        # Import the common classes
        from workbench.scenarios.common import ConflictExpectation, ScenarioDefinition

        # Convert YAML data to ConflictExpectation objects
        conflict_expectations = []
        for conflict_data in yaml_data.get('expected_conflicts', []):
            conflict_expectations.append(ConflictExpectation(**conflict_data))

        # Create ScenarioDefinition
        scenario = ScenarioDefinition(
            name=yaml_data['name'],
            description=yaml_data['description'],
            input_files=yaml_data['input_files'],
            expected_conflicts=conflict_expectations,
            expected_resolution=yaml_data.get('expected_resolution', ''),
            validation_checks=yaml_data.get('validation_checks', [])
        )

        return scenario

    except Exception as e:
        raise ValueError(f"Failed to load YAML scenario: {e}")


def generate_python_scenario_file(scenario, output_path: Path):
    """Generate a Python scenario file from a ScenarioDefinition."""

    # Generate the Python code
    python_code = f'''"""
{scenario.description}
"""

from .common import ConflictExpectation, ScenarioDefinition


def get_scenario() -> ScenarioDefinition:
    """Get the {scenario.name} scenario definition."""

    return ScenarioDefinition(
        name="{scenario.name}",
        description="{scenario.description}",
        input_files={repr(scenario.input_files)},
        expected_conflicts=[
{chr(10).join(f"            ConflictExpectation(**{repr(conflict.model_dump())})," for conflict in scenario.expected_conflicts)}
        ],
        expected_resolution="{scenario.expected_resolution}",
        validation_checks={repr(scenario.validation_checks)}
    )
'''

    with output_path.open('w') as f:
        f.write(python_code)


def add_scenario_command(args):
    """Add a new scenario from YAML/JSON."""
    console = Console()

    if not args.from_yaml:
        console.print("[yellow]⚠️  Please provide a YAML file with --from-yaml[/yellow]")
        console.print("Example usage: python workbench_cli.py add-scenario --from-yaml scenario.yaml")
        return

    yaml_path = Path(args.from_yaml)
    if not yaml_path.exists():
        console.print(f"[red]Error: YAML file not found: {yaml_path}[/red]")
        return

    try:
        # Load the YAML scenario
        scenario = load_yaml_scenario(yaml_path)

        # Generate Python scenario file
        scenarios_dir = Path(__file__).parent / "scenarios"
        output_path = scenarios_dir / f"{scenario.name}.py"

        if output_path.exists():
            console.print(f"[red]Error: Scenario file already exists: {output_path}[/red]")
            return

        generate_python_scenario_file(scenario, output_path)

        console.print(f"[green]✅ Successfully created scenario: {scenario.name}[/green]")
        console.print(f"File created: {output_path}")
        console.print("You can now run it with:")
        console.print(f"python workbench_cli.py run --scenario {scenario.name}")

    except Exception as e:
        console.print(f"[red]Error creating scenario: {e}[/red]")


async def main():
    """Main entry point for the workbench CLI."""
    parser = argparse.ArgumentParser(description="Context Mixer Conflict Detection Workbench")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Get available scenarios dynamically
    available_scenarios = list(discover_scenarios().keys())

    # Run command
    run_parser = subparsers.add_parser('run', help='Run scenarios')
    run_parser.add_argument(
        '--scenario',
        help='Run a specific scenario (default: run all scenarios)',
        choices=available_scenarios
    )
    run_parser.set_defaults(func=run_command)

    # List scenarios command
    list_parser = subparsers.add_parser('list-scenarios', help='List all available scenarios')
    list_parser.set_defaults(func=list_scenarios_command)

    # Validate scenario command
    validate_parser = subparsers.add_parser('validate-scenario', help='Validate a specific scenario')
    validate_parser.add_argument('scenario_name', help='Name of the scenario to validate')
    validate_parser.set_defaults(func=validate_scenario_command)

    # Add scenario command (placeholder)
    add_parser = subparsers.add_parser('add-scenario', help='Add a new scenario from YAML/JSON')
    add_parser.add_argument('--from-yaml', help='Path to YAML scenario definition')
    add_parser.set_defaults(func=add_scenario_command)

    args = parser.parse_args()

    # If no command specified, default to run
    if not args.command:
        args.command = 'run'
        args.scenario = None
        args.func = run_command

    # Handle async commands
    if args.command == 'run':
        await args.func(args)
    else:
        args.func(args)


if __name__ == "__main__":
    asyncio.run(main())
