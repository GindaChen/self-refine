

# MAX_ATTEMPTS=6
# NUM_QUESTIONS=1319
# ENTROPY_CUTOFF=0.5
# SKIP_GEN=0
# SKIP_EVAL=0
# RUNS=1

# while [[ "$#" -gt 0 ]]; do
#     case $1 in
#         --max_attempts) MAX_ATTEMPTS="$2"; shift;;
#         --num_questions) NUM_QUESTIONS="$2"; shift  ;;
#         --entropy_cutoff) ENTROPY_CUTOFF="$2"; shift;;
#         --skip_gen) SKIP_GEN=1;;
#         --skip_eval) SKIP_EVAL=1;;
#         --runs) RUNS="$2"; shift;;
#         *) echo "Unknown parameter passed: $1"; exit 1 ;;
#     esac
#     shift
# done

# set -e

# echo "Running $RUNS runs"
# for i in $(seq 0 $((RUNS))); do
#     echo "Running run $i"
#     bash run_gsm8k.sh \
#     --max_attempts $MAX_ATTEMPTS \
#     --num_questions $NUM_QUESTIONS \
#     --entropy_cutoff $ENTROPY_CUTOFF \
#     $([ $SKIP_GEN -eq 1 ] && echo "--skip_gen" || echo "") \
#     $([ $SKIP_EVAL -eq 1 ] && echo "--skip_eval" || echo "") \
#     --run_id $i
# done


MAX_ATTEMPTS=6
NUM_QUESTIONS=1319


start_run_id=0
end_run_id=9

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --start_run_id) start_run_id="$2"; shift ;;
        --end_run_id) end_run_id="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


# for entropy_cutoff in $(seq -1 0.1 0.05); do
for run_id in $(seq $start_run_id $end_run_id); do    
    for entropy_cutoff in 0.05; do
        echo "Running run $run_id with entropy cutoff $entropy_cutoff"
        OUTFILE=gsm8k_outputs.r${run_id}.e${entropy_cutoff}.jsonl
        EVALFILE=gsm8k_outputs.r${run_id}.e${entropy_cutoff}.eval.jsonl

        # python -u src/gsm/run_parallel.py  \
        # --max_attempts $MAX_ATTEMPTS \
        # --outfile $OUTFILE \
        # --num_questions $NUM_QUESTIONS

        python -u src/gsm/gsm_selfref_eval.py \
        --n_attempts $MAX_ATTEMPTS \
        --path $OUTFILE \
        --output_path $EVALFILE \
        --num_gsm $NUM_QUESTIONS \
        --entropy_cutoff $entropy_cutoff &
    done
done

wait