#!/bin/bash

MODELS=("GPT5Mini")
PROMPTS=("FewShot" "FewShot_Reverse" "FewShot_Random")

for LLM in "${MODELS[@]}"; do
    for PROMPT in "${PROMPTS[@]}"; do
        OUT_DIR="results_prompt_modes/${LLM}-${PROMPT}"
        mkdir -p "$OUT_DIR"

        nohup ./LLMsForEduQG.sh \
            -i datasets/SciQ_100_valid.csv \
            -o "$OUT_DIR" \
            -m "$LLM" \
            -p "$PROMPT" \
            > "$OUT_DIR/${LLM}-${PROMPT}_all.out" \
            2> "$OUT_DIR/${LLM}-${PROMPT}_all_err.out" &
    done
done