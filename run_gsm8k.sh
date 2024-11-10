MAX_ATTEMPTS=5
NUM_QUESTIONS=1319
ENTROPY_CUTOFF=0.5

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --max_attempts) MAX_ATTEMPTS="$2"; shift ;;
        --num_questions) NUM_QUESTIONS="$2"; shift ;;
        --entropy_cutoff) ENTROPY_CUTOFF="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

set -e

bash run_gsm8k_output.sh \
--max_attempts $MAX_ATTEMPTS \
--num_questions $NUM_QUESTIONS \
--outfile gsm8k_outputs.jsonl

bash run_gsm8k_eval.sh \
--max_attempts $MAX_ATTEMPTS \
--num_questions $NUM_QUESTIONS \
--entropy_cutoff $ENTROPY_CUTOFF \
--outfile gsm8k_outputs.jsonl \
--eval_file gsm8k_outputs_e${ENTROPY_CUTOFF}.jsonl
