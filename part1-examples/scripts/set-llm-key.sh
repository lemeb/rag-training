#!/bin/bash

# Get git root directory
git_root=$(git rev-parse --show-toplevel)

# Read KEY value from .env in git root
key_value=$(grep '^OPENAI_API_KEY=' "$git_root/.env" | cut -d'=' -f2)

# Pass to uv run command
echo "$key_value" | xargs uvx llm keys set openai --value
