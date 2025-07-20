#!/bin/bash

# Pre-commit Hook Installation Instructions:
# 1. Make this script executable: chmod +x commit-hook.sh
# 2. Copy this script to your git hooks directory: cp commit-hook.sh .git/hooks/pre-commit
# 3. Alternatively, create a symlink: ln -sf commit-hook.sh .git/hooks/pre-commit
#
# This hook runs the same validation checks as our CI pipeline:
# - flake8 linting for syntax errors and code style
# - pytest for running all tests
#
# If any check fails, the commit will be blocked.

set -e

echo "Running pre-commit validation..."

# Check if we're in a Python virtual environment, if not try to activate one
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        echo "Activating virtual environment..."
        source .venv/bin/activate
    elif [[ -f "venv/bin/activate" ]]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    fi
fi

echo "Running flake8..."
 # stop the validation if there are Python syntax errors or undefined names
 flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
 if [ $? -ne 0 ]; then
     echo "flake8 found critical errors. Commit aborted."
     exit 1
 fi
 # exit-zero treats all errors as warnings
 flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --ignore=F401

 echo "Running pytest..."
 python -m pytest
 exit_code=$?
 exit $exit_code