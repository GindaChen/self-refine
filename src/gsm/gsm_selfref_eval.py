from importlib import reload
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager
import signal
from glob import glob
import os
import sys


sys.path.append(".")

# from https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"block timedout after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

def read_json(path):
    import json
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    
    task_df = pd.DataFrame(rows)
    return task_df

def evaluate_code_prompt(path, output_path, num_gsm: int = 1319, n_attempts: int = 5, entropy_cutoff: float = 0):
    data = read_json(path)
    if "question" not in data.columns:
        data["question"] = data["input"]
    if "answer" not in data.columns:
        data["answer"] = data["target"]

    attempt_to_acc = []
    total_prompt_tokens = []
    total_output_tokens = []

    reports = []  # Step 1
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Evaluate GSM8k"):
        # if idx < 20:
        #     continue
        # if idx > 10:
        #     break
        attempt_to_acc_ = {i: 0 for i in range(n_attempts)}
        attempt_to_acc_["question"] = row["question"]
        solutions = []
        if row["run_logs"] is None:
            continue
        for _, log in enumerate(row["run_logs"]):
            solutions.append(log["solution_curr"])
            if 'entropy' in log:
                entropy = log['entropy']
                if entropy < entropy_cutoff:
                    break
            
        solutions.append(row["run_logs"][-1]["solution_fixed"])
        
        feedback = [rec["feedback"] for rec in row["run_logs"]]
        prompt_tokens = [rec["total_prompt_tokens_at_attempt"] for rec in row["run_logs"]]
        output_tokens = [rec["total_output_tokens_at_attempt"] for rec in row["run_logs"]]
        total_prompt_tokens.append(prompt_tokens)
        total_output_tokens.append(output_tokens)


        prev_accuracy = 0
        for iter_idx, soln in enumerate(solutions):
            soln = soln.split("\n\n\n")[0].strip() + "\n"
            soln = soln.replace("The answer is", "").strip() + "\n"
            # os.system("rm -rf __pycache__")
            # os.system("rm -f temp_result.pyc")

            with open("temp_result.py", "w+") as f:
                f.write(soln)
            try:
                with timeout(3):
                    import importlib.util
                    import pathlib
                    
                    # Remove the module from sys.modules if it exists
                    if 'temp_result' in sys.modules:
                        del sys.modules['temp_result']
                    
                    # Force load from .py file
                    spec = importlib.util.spec_from_file_location("temp_result", "temp_result.py")
                    temp_result = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(temp_result)

                    # import temp_result
                    # temp_result.__file__ = "temp_result.py"
                    # reload(temp_result)

                    correct_solution = str(row["answer"])
                    exec(soln)
                    result = str(temp_result.solution())
                
                is_corr = check_corr(result, correct_solution)

                is_corr = int(is_corr)
                # Step 2
                
                if iter_idx > 0 and is_corr == 1 and prev_accuracy == 0:
                    report = {
                        "previous_solution": solutions[iter_idx - 1],
                        "feedback": feedback[iter_idx - 1],
                        "next_solution": solutions[iter_idx],
                    }
                    reports.append(report)  # Step 3
                if is_corr == 1:
                    for j in range(iter_idx, n_attempts):
                        attempt_to_acc_[j] = 1
                    break
                attempt_to_acc_[iter_idx] = 0
                prev_accuracy = is_corr

                # total_prompt_tokens.append(row["total_prompt_tokens_at_attempt"])
                # total_output_tokens.append(row["total_output_tokens_at_attempt"])
            except Exception as e:
                import traceback
                traceback.print_exc()
                # print("Error while executin code:")
                # print(soln)
                continue

        attempt_to_acc.append(attempt_to_acc_)

    total_prompt_tokens_df = pd.DataFrame(total_prompt_tokens)
    total_prompt_tokens_per_iter = total_prompt_tokens_df.sum(axis=0).apply(int).tolist()
    
    total_output_tokens_df = pd.DataFrame(total_output_tokens)
    total_output_tokens_per_iter = total_output_tokens_df.sum(axis=0).apply(int).tolist()

    import numpy as np
    total_tokens_per_iter = [p + o for (p, o) in zip(total_prompt_tokens_per_iter, total_output_tokens_per_iter)]
    total_tokens_per_iter_cumsum = np.cumsum(total_tokens_per_iter).tolist()
    
    df = pd.DataFrame(attempt_to_acc)
    df.to_json(output_path, orient="records", lines=True)
    # breakpoint()

    report_file = f"{output_path}.reports.txt"
    print_reports(reports, report_file, df, num_gsm, n_attempts)  # Step 4

    
    data = {
        'Attempt': list(range(n_attempts)),
        'Prompt Tokens': total_prompt_tokens_per_iter,
        'Output Tokens': total_output_tokens_per_iter,
        'Total Tokens': total_tokens_per_iter,
        'Cumulative Total Tokens': total_tokens_per_iter_cumsum,
        'Accuracy': [df[i].sum() / num_gsm for i in range(n_attempts)],
        'Correct Questions': [df[i].sum() for i in range(n_attempts)],
        'Total Questions': [num_gsm for i in range(n_attempts)]
    }

    breakpoint()
    stat_df = pd.DataFrame(data)
    print(stat_df.to_markdown())
    stat_df.to_csv(f"{output_path}.stats.csv")

    # print(attempt_to_acc)
    for i in range(n_attempts):
        print(f"Accuracy at attempt {i} = {df[i].sum() / num_gsm:.2%} ({df[i].sum()}/{num_gsm})")

    return reports

# Step 4
def print_reports(reports, report_file, df, num_gsm, n_attempts):
    with open(report_file, "w") as f:
        for i in range(n_attempts):
            print(f"Accuracy at attempt {i} = {df[i].sum() / num_gsm:.2%} ({df[i].sum()}/{num_gsm})", file=f)

        for i, report in enumerate(reports):
            f.write(f"Report {i + 1}:\n")
            f.write("\nPrevious solution:\n")
            f.write(report["previous_solution"])
            f.write("\n\nFeedback:\n")
            f.write(report["feedback"])
            f.write("\n\nNext solution:\n")
            f.write(report["next_solution"])
            f.write("\n\n" + "=" * 80 + "\n\n")


def check_corr(result: float, correct_solution: float, tol: float = 1e-3):
    if result.strip() == correct_solution.strip():
        return 1
    try:
        result = float(result.strip())
        correct_solution = float(correct_solution.strip())
        return abs(result - correct_solution) < tol
    except:
        return 0



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/quco/quco_test.jsonl")
    parser.add_argument("--num_gsm", type=int, default=1319)
    parser.add_argument("--n_attempts", type=int, default=5)
    parser.add_argument("--output_path", type=str, default="data/quco/quco_test.jsonl")
    parser.add_argument("--entropy_cutoff", type=float, default=0)
    args = parser.parse_args()
    
    evaluate_code_prompt(args.path, args.output_path, args.num_gsm, args.n_attempts, args.entropy_cutoff)
