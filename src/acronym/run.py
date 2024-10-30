import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import tqdm
import os    
import sys
import pathlib

from src.acronym.task_init import AcronymGenTaskInit
from src.acronym.task_iterate import AcronymGenTaskIterate
from src.acronym.feedback import AcronymGenFeedback
from src.utils import retry_parse_fail_prone_cmd

CODEX = "code-davinci-002"
GPT3 = "text-davinci-003"
CHAT_GPT = "gpt-3.5-turbo"
GPT4 = "gpt-4"


ENGINE = CHAT_GPT


import math
from typing import List

def length_normalized_entropy(logprobs: List[float]) -> float:
    return sum(logprobs) / len(logprobs)
    

@retry_parse_fail_prone_cmd
def iterative_acronym(title: str, max_attempts: int) -> str:
    
    # initialize all the required components
    
    # generation of the first acronym
    task_init = AcronymGenTaskInit(engine=ENGINE, prompt_examples="data/prompt/acronym/init.jsonl")
    
    # getting feedback
    task_feedback = AcronymGenFeedback(engine=ENGINE, prompt_examples="data/prompt/acronym/feedback.jsonl")

    # iteratively improving the acronym
    task_iterate = AcronymGenTaskIterate(engine=ENGINE, prompt_examples="data/prompt/acronym/feedback.jsonl")
    
    
    # Initialize the task

    n_attempts = 0
    
    # print(f"{n_attempts} INIT> {title}")
    acronyms_to_scores = dict()
    
    all_acronyms_to_scores = dict()
    all_acronyms_to_scores_logs = []
    best_score_so_far = 0

    token_logprobs = []

    progress_bar = tqdm.tqdm(total=max_attempts, desc=f"Generating acronyms {title}")
    while n_attempts < max_attempts:

        if n_attempts == 0:
            acronym, logprobs = task_init(title=title)
            token_logprobs.extend(logprobs)
        else:
            new_title, acronym, logprobs = task_iterate(acronyms_to_scores=acronyms_to_scores)
            token_logprobs.extend(logprobs)
            title = new_title

        before_feedback_entropy = length_normalized_entropy(token_logprobs)
        scores, logprobs = task_feedback(title=title, acronym=acronym)
        token_logprobs.extend(logprobs)
        after_feedback_entropy = length_normalized_entropy(token_logprobs)
        
        # extract expression "Total score: 22/25" from scores
        total_score = re.search(r"Total score: (\d+)/(\d+)", scores).group(0)
        total_score = int(total_score.split(":")[1].strip().split("/")[0])
        
        round_result = {
            "acronym": acronym,
            "scores": scores,
            "total_score": total_score,
            "title": title,
            "n_attempts": n_attempts,
            "token_logprobs": token_logprobs,
            "before_feedback_entropy": before_feedback_entropy,
            "after_feedback_entropy": after_feedback_entropy,
        }
        all_acronyms_to_scores[acronym] = round_result
        all_acronyms_to_scores_logs.append(round_result)

        # print(f"{n_attempts} GEN> {acronym} TITLE> {title}")

        # print(f"{n_attempts} SCORES> {scores}")
        if total_score >= 0:  # only iterate over things that are improving
            best_score_so_far = total_score
            acronyms_to_scores[acronym] = (title, scores)
        else:
            print(f"Score of {acronym} is {total_score}, which is less than the current best of {best_score_so_far}")
        # print(f"{n_attempts} SCORE> {total_score}")

        n_attempts += 1
        progress_bar.update(1)

    # return all_acronyms_to_scores
    return all_acronyms_to_scores_logs


def _parse_results(title: str, ground_truth_acronym, max_attempts: int) -> str:
    try:
        results = iterative_acronym(title=title, max_attempts=max_attempts)
        for d in results:
            d['ground_truth'] = ground_truth_acronym
        return results
    except Exception as e:
        return []


def run_over_titles(titles_file: str, max_attempts: int, outfile: str):
    data = pd.read_csv(titles_file, sep="\t")
    data = data.head(1)

    results = []
    with ThreadPoolExecutor(max_workers=256) as executor:
    # with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(
                _parse_results, 
                title, ground_truth_acronym, max_attempts
            ) 
            for title, ground_truth_acronym in zip(data['title'], data['acronym'])
        ]
        
        progress_bar = tqdm.tqdm(total=len(futures), desc="Processing titles")
        for future in futures:
            result = future.result()
            progress_bar.update(1)
            if result is not None:
                results.extend(result)
            pass    

    result_data = pd.DataFrame(results)
    # result_data.drop(columns=['scores'], inplace=True)
    result_data = result_data[['n_attempts', 'acronym', 'ground_truth', 'title', 'total_score']]
    
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w+") as f:
        result_data.to_csv(f, sep=",", index=False)



if __name__ == "__main__":
    default_acronym_dir_file = pathlib.Path(__file__).parent.parent.parent / "data/tasks/acronyms" / "acronyms.tsv"
    title = sys.argv[1]  # Light Amplification by Stimulated Emission of Radiation
    if len(sys.argv) > 2:
        run_over_titles(
            titles_file=sys.argv[1],
            max_attempts=int(sys.argv[2]),
            outfile=sys.argv[3],
        )
    else:
        max_attempts = 5
        all_acronyms_to_scores = iterative_acronym(
            title=title,
            max_attempts=max_attempts,
        )
        
        res = []
        for acronym, scores in all_acronyms_to_scores.items():
            res.append(f"{acronym} [score: {scores['total_score']}] \n {scores['scores']}")
        print("\n ------ \n ".join(res))

