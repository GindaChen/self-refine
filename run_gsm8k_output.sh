MAX_ATTEMPTS=5
NUM_QUESTIONS=1319
OUTFILE=gsm8k_outputs.jsonl

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --max_attempts) MAX_ATTEMPTS="$2"; shift ;;
        --num_questions) NUM_QUESTIONS="$2"; shift ;;
        # --entropy_cutoff) ENTROPY_CUTOFF="$2"; shift ;;
        --outfile) OUTFILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


# ENTROPY_CUTOFF=0.5

# python -u src/gsm/run_parallel.py  \

# python -u src/gsm/run_parallel_sglang.py  \

set -xe
# python -u src/gsm/run_parallel.py  \
python -u src/gsm/run_parallel_sglang.py  \
--max_attempts $MAX_ATTEMPTS \
--outfile $OUTFILE \
--num_questions $NUM_QUESTIONS
set +x

# python -u src/gsm/gsm_selfref_eval.py \
# --path  gsm8k_outputs.jsonl \
# --output_path gsm8k_eval_e${ENTROPY_CUTOFF}.jsonl \
# --num_gsm $NUM_QUESTIONS \
# --entropy_cutoff $ENTROPY_CUTOFF

# echo "Result running GSM8k with original self-refine with entropy cutoff $ENTROPY_CUTOFF"