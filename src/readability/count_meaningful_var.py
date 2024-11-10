import json
from tqdm import tqdm
from argparse import ArgumentParser

from src.readability.utils import call_gpt
from src.readability.prompts import COUNT_VAR_PROMPT

def count_meaningful_vars(code):
    if 'Fixed Code:' in code:
        code = code.split('Fixed Code:')[1]
    
    code = code.strip()
    prompt = COUNT_VAR_PROMPT.format(code=code)
    result = call_gpt(prompt, model='code-davinci-002', max_tokens=256, stop='\n\n\n')[0]
    
    result = result.strip().splitlines()
    num_vars = len(result)
    num_random_vars = sum([1 for line in result if line.endswith('- random')])
    num_meaningful_vars = num_vars - num_random_vars
        
    return num_meaningful_vars, num_meaningful_vars / num_vars, result


def get_meaningful_var(line):
    data = json.loads(line)
    original_code = data['original_code']
    updated_codes = [x['updated_code'] for x in data['updates']]

    score_counter = [[] for _ in range(len(updated_codes) + 1)]

    data['update_meaningful_var_count'] = []
    data['update_meaningful_var_ratio'] = []

    for i, code in enumerate([original_code] + updated_codes):
        num_meaningful_vars, ratio, gpt_gen = -1, -1, None
        if code:
            num_meaningful_vars, meaningful_var_ratio, gpt_gen = count_meaningful_vars(code)
        data['update_meaningful_var_count'].append(num_meaningful_vars)
        data['update_meaningful_var_ratio'].append(meaningful_var_ratio)
        
        score_counter[i].append(max(0, meaningful_var_ratio))
    return data, score_counter


from concurrent.futures import ThreadPoolExecutor

def main_multithread():
    parser = ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    output = args.file[:-6] + '_var_count.jsonl'

    with open(output, 'w') as fout:
        input_lines = open(args.file).readlines()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for line in tqdm(input_lines):
                futures.append(executor.submit(get_meaningful_var, line))
            
            score_counters = []
            for future in tqdm(futures, total=len(futures)):
                data, score_counter = future.result()
                score_counters.append(score_counter)
                fout.write(json.dumps(data) + '\n')
            
            # Initialize combined score counter
            combined_scores = [[] for _ in range(len(score_counters[0]))]
            for score_counter in score_counters:
                for i, scores in enumerate(score_counter):
                    for score in scores:
                        combined_scores[i].append(score)
            
            for i, scores in enumerate(combined_scores):
                print(f'Update {i}: avg meaningful var ratio {sum(scores) / len(scores)}')
            
    return


def main():
    parser = ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    output = args.file[:-6] + '_var_count.jsonl'
    score_counter = None

    with open(output, 'w') as fout:
        input_lines = open(args.file).readlines()
        for line in tqdm(input_lines):
            data = json.loads(line)
            original_code = data['original_code']
            updated_codes = [x['updated_code'] for x in data['updates']]
            
            if score_counter is None:
                score_counter = [[] for _ in range(len(updated_codes) + 1)]
            
            data['update_meaningful_var_count'] = []
            data['update_meaningful_var_ratio'] = []
            
            for i, code in enumerate([original_code] + updated_codes):
                num_meaningful_vars, ratio, gpt_gen = -1, -1, None
                if code:
                    num_meaningful_vars, meaningful_var_ratio, gpt_gen = count_meaningful_vars(code)
                data['update_meaningful_var_count'].append(num_meaningful_vars)
                data['update_meaningful_var_ratio'].append(meaningful_var_ratio)
                
                score_counter[i].append(max(0, meaningful_var_ratio))
            fout.write(json.dumps(data) + '\n')
            
    for i, scores in enumerate(score_counter):
        print(f'Update {i}: avg meaningful var ratio {sum(scores) / len(scores)}')
        

if __name__ == '__main__':
    # main()
    main_multithread()