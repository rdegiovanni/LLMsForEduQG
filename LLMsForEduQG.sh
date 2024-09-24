#!/bin/bash
export OPENAI_API_KEY=
export API_KEY_HUGGINGFACE=

source venv/bin/activate

python3 LLMsForEduQG/Main.py "$@"