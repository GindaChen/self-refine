MAX_ATTEMPTS=5
NUM_QUESTIONS=1319
ENTROPY_CUTOFF=0.5
SKIP_GEN=0
SKIP_EVAL=0
RUN_ID=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --max_attempts) MAX_ATTEMPTS="$2"; shift;;
        --num_questions) NUM_QUESTIONS="$2"; shift ;;
        --entropy_cutoff) ENTROPY_CUTOFF="$2"; shift;;
        --run_id) RUN_ID="$2"; shift;;
        --skip_gen) SKIP_GEN=1;;
        --skip_eval) SKIP_EVAL=1;;
        *) echo "run_gsm8k.sh: Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

set -e

OUTPUT_FILE=gsm8k_outputs.r${RUN_ID}.jsonl
if [ $SKIP_GEN -eq 0 ]; then
    bash run_gsm8k_output.sh \
    --max_attempts $MAX_ATTEMPTS \
    --num_questions $NUM_QUESTIONS \
    --outfile $OUTPUT_FILE
fi

if [ $SKIP_EVAL -eq 0 ]; then
    bash run_gsm8k_eval.sh \
    --max_attempts $MAX_ATTEMPTS \
    --num_questions $NUM_QUESTIONS \
    --entropy_cutoff $ENTROPY_CUTOFF \
    --outfile $OUTPUT_FILE \
    --eval_file ${OUTPUT_FILE}.e${ENTROPY_CUTOFF}.jsonl
fi
