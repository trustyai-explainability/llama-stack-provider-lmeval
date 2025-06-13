#!/bin/sh

# This script is used to prepare the virtual environment for the Llama Stack LMEval Provider

uv install -e .
source .venv/bin/activate
llama stack run run.yaml
