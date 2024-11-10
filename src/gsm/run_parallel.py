import pandas as pd
from tqdm import tqdm


from src.gsm.task_init import GSMInit
from src.gsm.feedback import GSMFeedback

from src.utils import retry_parse_fail_prone_cmd

CODEX = "code-davinci-002"
# GPT3 = "text-davinci-003"
ENGINE = CODEX



@retry_parse_fail_prone_cmd
def iterative_gsm(question: str, max_attempts: int, feedback_type: str, temperature: float, entropy_cutoff: float):
    # print(f"iterative_gsm {question} {max_attempts} {feedback_type} {temperature}")

    # initialize all the required components

    # generation of the first fast version
    task_init = GSMInit(engine=ENGINE, prompt_examples="data/prompt/gsm/init.txt", temperature=temperature)

    # getting feedback
    if feedback_type == "naive":
        raise NotImplementedError
    else:
        task_feedback = GSMFeedback(engine=ENGINE, prompt_examples="data/prompt/gsm/feedback.txt", temperature=0.7)

    n_attempts = 0
    log = []

    total_prompt_tokens = 0
    total_output_tokens = 0
    while n_attempts < max_attempts:
        if n_attempts == 0:
            solution, init_tokens = task_init(solution=question)
            total_prompt_tokens += init_tokens["prompt_tokens"]
            total_output_tokens += init_tokens["output_tokens"]
        
        fb_and_maybe_soln = task_feedback(solution=solution)
        total_prompt_tokens += fb_and_maybe_soln["prompt_tokens"]
        total_output_tokens += fb_and_maybe_soln["output_tokens"]

        # entropy = fb_and_maybe_soln["entropy"]
        
        item = {
            "attempt": n_attempts, 
            "solution_curr": solution, 
            "solution_fixed": fb_and_maybe_soln["solution"], 
            "feedback": fb_and_maybe_soln["feedback"], 
            "total_prompt_tokens_at_attempt": total_prompt_tokens,
            "total_output_tokens_at_attempt": total_output_tokens,
            # "entropy": entropy,
        }
        log.append(item)

        if "it is correct" in fb_and_maybe_soln["feedback"].lower():
            break

        solution = fb_and_maybe_soln["solution"]
        n_attempts += 1

    # print(f"iterative_gsm {question} {max_attempts} {feedback_type} {temperature} done")
    return log


def fix_gsm(gsm_task_file: str, max_attempts: int, outfile: str, new_out_file: str, feedback_type: str, temperature: float, num_questions: int = None, entropy_cutoff:float = 0):


    slow_programs_df = pd.read_json(gsm_task_file, lines=True, orient="records")

    if num_questions is None:
        num_questions = len(slow_programs_df)

    slow_programs_df = slow_programs_df.head(num_questions)
    
    slow_programs_df["run_logs"] = None
    futures = []
    results = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1319) as executor:
        for i, row in tqdm(slow_programs_df.iterrows(), total=len(slow_programs_df)):
            future = executor.submit(iterative_gsm, question=row["input"], max_attempts=max_attempts, feedback_type=feedback_type, temperature=temperature, entropy_cutoff=entropy_cutoff)
            futures.append(future)

        
        for i, row in tqdm(slow_programs_df.iterrows(), total=len(slow_programs_df)):
            row_copy = row.to_dict()
            try:
                future = futures[i]
                run_logs = future.result()
                row_copy["run_logs"] = run_logs
                row_copy["generated_answer_ours"] = run_logs[-1]["solution_fixed"]
                row_copy["generated_answer_direct"] = run_logs[0]["solution_curr"]
                results.append(row_copy)
                # if i % 10 == 0:
                #     pd.DataFrame(results).to_json(outfile + f".{i}.jsonl", orient="records", lines=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                pass
    
    # breakpoint()
    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    pd.DataFrame(results).to_json(new_out_file, orient="records", lines=True)
    return results


def test():
    import json

    
    with open("/tmp/debug_gsm.jsonl", "w") as fout:
        fout.write(json.dumps({"input": "Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup."}))
        
    logs = fix_gsm(
        gsm_task_file="/tmp/debug_gsm.jsonl", max_attempts=3, outfile="/tmp/test.jsonl", feedback_type="rich", temperature=0.0
    )
    for i, log in enumerate(logs):
        print(log["generated_answer_ours"])
        print(log["generated_answer_direct"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and sys.argv[1] == "test":
        test()
        exit(0)
    
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--gsm_task_file", type=str, default="data/tasks/gsm/gsm.jsonl")
    args.add_argument("--max_attempts", type=int, default=4)
    args.add_argument("--outfile", type=str, default="data/tasks/gsm/gsm_outputs.jsonl")
    args.add_argument("--feedback_type", type=str, default="rich")
    args.add_argument("--temperature", type=float, default=0.0)
    args.add_argument("--num_questions", type=int, default=None)
    args.add_argument("--entropy_cutoff", type=float, default=0)
    args = args.parse_args()
    new_out_file = args.outfile
    args.outfile = f"{args.outfile}.fb_{args.feedback_type}.temp_{args.temperature}.engine_{ENGINE}.jsonl"
    fix_gsm(
        gsm_task_file=args.gsm_task_file, max_attempts=args.max_attempts, outfile=args.outfile, new_out_file=new_out_file, 
        feedback_type=args.feedback_type, temperature=args.temperature, num_questions=args.num_questions, 
        entropy_cutoff=args.entropy_cutoff,
    )
