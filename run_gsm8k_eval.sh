

MAX_ATTEMPTS=5
NUM_QUESTIONS=1319
ENTROPY_CUTOFF=0.5
OUTFILE=gsm8k_outputs.jsonl
EVALFILE=gsm8k_outputs_e${ENTROPY_CUTOFF}.jsonl

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --max_attempts) MAX_ATTEMPTS="$2"; shift ;;
        --num_questions) NUM_QUESTIONS="$2"; shift ;;
        --entropy_cutoff) ENTROPY_CUTOFF="$2"; shift ;;
        --outfile) OUTFILE="$2"; shift ;;
        --eval_file) EVALFILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

set -xe
python -u src/gsm/gsm_selfref_eval.py \
--n_attempts $MAX_ATTEMPTS \
--path $OUTFILE \
--output_path $EVALFILE \
--num_gsm $NUM_QUESTIONS \
--entropy_cutoff $ENTROPY_CUTOFF
set +x

echo "Result running GSM8k with original self-refine with entropy cutoff $ENTROPY_CUTOFF"