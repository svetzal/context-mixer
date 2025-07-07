#!/bin/sh
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