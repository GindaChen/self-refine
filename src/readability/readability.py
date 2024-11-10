import json
import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from argparse import ArgumentParser

from src.readability.utils import call_gpt
from src.readability.prompts import PROMPT_CRITIQUE, PROMPT_FIX

ROUNDS = 5
FILE_PATH = 'data/tasks/codeclean/code_readability/codenet-python-train.jsonl'

def fix_readability(example):
    rounds = []
    code = example['input']
    code = code.replace('\n\n', '\n')
    for round_number in range(ROUNDS):
        prompt = PROMPT_CRITIQUE.format(code=code)
        suggestion = call_gpt(prompt, temperature=0.0)[0]
        prompt = PROMPT_FIX.format(code=code, suggestion=suggestion)
        code = call_gpt(prompt)[0].strip()
        rounds.append({'suggestion': suggestion, 'updated_code': code})
    
    result_item = {
        'original_code': example['input'],
        'updates': rounds,
    }
    return result_item


def main():
    parser = ArgumentParser()
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--file', type=str, default=FILE_PATH)
    parser.add_argument('--output', type=str, default='readability.jsonl')
    args = parser.parse_args()
    print(args)
    
    N = args.n

    with open(args.file, 'r') as f:
        examples = [json.loads(line) for line in f.readlines()]

    if N is not None:
        N = min(N, len(examples))
    
    examples = examples[:N]
    
    # Create thread pool
    with ThreadPoolExecutor(max_workers=N) as executor:
        # Submit all examples to thread pool and wrap with tqdm
        futures = []
        for example in examples:
            futures.append(executor.submit(fix_readability, example))
        
        with open(args.output, 'w+') as f:
            # Write results as they complete
            for future in tqdm(futures, total=len(futures)):
                result_item = future.result()
                f.write(json.dumps(result_item) + '\n')
    

if __name__ == '__main__':
    main()