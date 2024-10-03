#!/bin/bash


MODELS=""
PROMPTS=""

for LLM in $MODELS; do
#for ((K=0;K<=690;K=K+10)); do
	for PROMPT in PROMPTS; do
		OUT_DIR=results/$LLM-$PROMPT
		nohup ./LLMsForEduQG.sh -i datasets/SciQ_test.csv -o $OUT_DIR 1> $OUT_DIR/$LLM-$PROMPT_all.out 2> $OUT_DIR/$LLM-$PROMPT_all_err.out &
	done
done