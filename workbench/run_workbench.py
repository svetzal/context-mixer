#!/usr/bin/env python3
"""
Conflict Detection and Resolution Workbench

This script runs automated tests for conflict detection and resolution scenarios.
It provides a persistent testing environment that accumulates troublesome scenarios
and ensures we can always pass them all.
"""

import argparse
import asyncio
import logging
import os
import sys
import tempfile
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
from context_mixer.gateways.llm import LLMGateway
from context_mixer.commands.ingest import do_ingest
from workbench.automated_resolver import AutomatedConflictResolver
from workbench.scenarios.indentation_conflict import get_scenario as get_indentation_scenario
from workbench.scenarios.false_positive_naming import get_scenario as get_false_positive_scenario
from workbench.scenarios.internal_conflict import get_scenario as get_internal_scenario

# Import OpenAI gateway
from mojentic.llm.gateways import OpenAIGateway


class WorkbenchRunner:
    """Main workbench runner that executes conflict detection scenarios."""

    def __init__(self):
        """Initialize the workbench runner."""
        self.console = Console()
        self.automated_resolver = AutomatedConflictResolver(self.console)

        # Configure OpenAI gateway
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
            sys.exit(1)

        openai_gateway = OpenAIGateway(api_key=api_key)
        self.llm_gateway = LLMGateway(model="gpt-4o-mini", gateway=openai_gateway)

        # Available scenarios
        self.scenarios = {
            "indentation_conflict": get_indentation_scenario,
            "false_positive_naming": get_false_positive_scenario,
            "internal_conflict": get_internal_scenario,
        }

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
                "errors": []
            }

            try:
                # Ingest files one by one
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
                    results["passed"] = all(result["passed"] for result in results["validation_results"])
                else:
                    results["errors"].append("context.md was not created")

            except Exception as e:
                results["errors"].append(str(e))
                self.console.print(f"[red]Error running scenario: {e}[/red]")

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

    async def run_all_scenarios(self) -> Dict[str, Any]:
        """
        Run all available scenarios.

        Returns:
            Dictionary with overall results
        """
        self.console.print("[bold green]🚀 Starting Conflict Detection Workbench[/bold green]")
        self.console.print(f"Using OpenAI model: gpt-4o-mini")

        overall_results = {
            "total_scenarios": len(self.scenarios),
            "passed_scenarios": 0,
            "failed_scenarios": 0,
            "scenario_results": []
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

                    # Show errors
                    for error in result["errors"]:
                        self.console.print(f"  [red]• Error: {error}[/red]")

            except Exception as e:
                overall_results["failed_scenarios"] += 1
                self.console.print(f"[red]❌ {scenario_name} FAILED with exception: {e}[/red]")

        return overall_results

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the workbench results."""

        # Create summary table
        table = Table(title="Workbench Results Summary")
        table.add_column("Scenario", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Validations", style="dim")

        for result in results["scenario_results"]:
            status = "[green]PASSED[/green]" if result["passed"] else "[red]FAILED[/red]"
            validation_count = len(result["validation_results"])
            passed_validations = sum(1 for v in result["validation_results"] if v["passed"])
            validations = f"{passed_validations}/{validation_count}"

            table.add_row(result["scenario_name"], status, validations)

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

        summary_text = f"""
{status_text}

Passed: {passed}/{total}
Failed: {failed}/{total}
        """

        self.console.print(Panel(summary_text.strip(), title="Final Results", style=panel_style))


async def main():
    """Main entry point for the workbench."""
    parser = argparse.ArgumentParser(description="Run conflict detection workbench scenarios")
    parser.add_argument(
        "--scenario", 
        help="Run a specific scenario (default: run all scenarios)",
        choices=["indentation_conflict", "false_positive_naming", "internal_conflict"]
    )

    args = parser.parse_args()

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


if __name__ == "__main__":
    asyncio.run(main())
