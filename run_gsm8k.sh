set -xe

MAX_ATTEMPTS=5
NUM_QUESTIONS=1319


# python -u src/gsm/run_parallel.py  \

python -u src/gsm/run_parallel_sglang.py  \
--max_attempts $MAX_ATTEMPTS \
--outfile debug.jsonl \
--num_questions $NUM_QUESTIONS

python -u src/gsm/gsm_selfref_eval.py \
--path  debug.jsonl \
--output_path eval.jsonl \
--num_gsm $NUM_QUESTIONS

echo "Result running GSM8k with original self-refine"