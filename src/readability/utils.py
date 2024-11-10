import openai
import time


from prompt_lib.backends.openllm import AnyOpenAILLM

client = AnyOpenAILLM()

# GPT-3 API
def call_gpt(prompt, model='text-davinci-003', stop=None, temperature=0., top_p=1.0,
        max_tokens=1024, majority_at=None, **kwargs):
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 5
    
    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        requested_completions = min(num_completions_batch_size, num_completions - len(completions))
        ans = client(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=requested_completions,
            best_of=requested_completions,
            **kwargs,
        )
        completions.extend([choice['text'] for choice in ans['choices']])
        if len(completions) >= num_completions:
            return completions[:num_completions]
    raise RuntimeError('Failed to call GPT API')