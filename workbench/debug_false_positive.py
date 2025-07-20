import asyncio
import os
import tempfile
from pathlib import Path

from mojentic.llm.gateways import OpenAIGateway

from context_mixer.commands.ingest import do_ingest
from context_mixer.config import Config
from context_mixer.gateways.llm import LLMGateway
from workbench.automated_resolver import AutomatedConflictResolver
from workbench.scenarios.false_positive_naming import get_scenario


async def debug_false_positive():
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Setup LLM
    openai_gateway = OpenAIGateway(api_key=api_key)
    llm_gateway = LLMGateway(model="o4-mini", gateway=openai_gateway)
    
    # Get scenario
    scenario = get_scenario()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        library_path = temp_path / "library"
        
        # Create test files
        for filename, content in scenario.input_files.items():
            file_path = temp_path / filename
            file_path.write_text(content)
            print(f"Created {filename}:")
            print(content)
            print("---")
        
        # Setup config
        config = Config(library_path=library_path)
        resolver = AutomatedConflictResolver()
        
        # Ingest files
        for filename in scenario.input_files.keys():
            file_path = temp_path / filename
            print(f"Ingesting {filename}...")
            await do_ingest(
                console=None,
                config=config,
                llm_gateway=llm_gateway,
                path=file_path,
                project_id="workbench-test",
                project_name="Workbench Test",
                commit=False,
                detect_boundaries=True,
                resolver=resolver
            )
        
        # Check context.md
        context_file = library_path / "context.md"
        if context_file.exists():
            content = context_file.read_text()
            print("=== CONTEXT.MD CONTENT ===")
            print(content)
            print("=== END CONTEXT.MD ===")
            
            # Check validations
            print("\n=== VALIDATION CHECKS ===")
            for check in scenario.validation_checks:
                if check.startswith("should_contain:"):
                    expected_text = check[15:]
                    passed = expected_text in content
                    print(f"✓ {check}: {'PASS' if passed else 'FAIL'}")
                    if not passed:
                        print(f"  Looking for: '{expected_text}'")
        else:
            print("ERROR: context.md was not created")

if __name__ == "__main__":
    asyncio.run(debug_false_positive())