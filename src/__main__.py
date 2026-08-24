# from llm_sdk import Small_LLM_Model

# model = Small_LLM_Model()

# model.encode("What is the sum of 2 and 3?")


from llm_sdk import Small_LLM_Model
from pathlib import Path
from tokenizer import Tokenizer
from time import perf_counter

model = Small_LLM_Model()

# vocab_file = Path(model.get_path_to_vocab_file())
# merges_file = Path(model.get_path_to_merges_file())
tokenizer_file = Path(model.get_path_to_tokenizer_file())

# print(vocab_file)
# print(merges_file)
print(tokenizer_file)

tokenizer = Tokenizer(tokenizer_file)
# prompt = input("prompt: ")

# normalized_text = tokenizer._normalize(prompt)
# pre_tokenized_text = tokenizer._pre_tokenize(normalized_text)
# t = perf_counter()
# encoded = tokenizer.encode(prompt)
# my_encode_time = perf_counter() - t
# t = perf_counter()
# model_encoded = model.encode(prompt)[0].tolist()
# model_encode_time = perf_counter() - t
# t = perf_counter()
# decoded = tokenizer.decode(encoded)
# my_decode_time = perf_counter() - t
# t = perf_counter()
# model_decoded = model.decode(model_encoded)
# model_decode_time = perf_counter() - t

# print(f"original text:      {repr(prompt)}")
# print(f"normalized text:    {repr(normalized_text)}")
# print(f"pre_tokenized_text: {pre_tokenized_text}")
# print(f"my encoder:         {encoded} {my_encode_time:.5f}")
# print(f"model encoder:      {model_encoded} {model_encode_time:.5f}")
# print(f"my decoder:         {repr(decoded)} {my_decode_time:.5f}")
# print(f"model decoder:      {repr(model_decoded)} {model_decode_time:.5f}")

# exit()
functions_def_file = "data/input/functions_definition.json"
prompts_file = "data/input/function_calling_tests.json"

import json
with open(functions_def_file) as f:
    functions_defs = json.dumps(json.load(f))
with open(prompts_file) as f:
    prompts = json.load(f)


prompt_text = f"""
You are a function-calling model.

Here are the available function definitions:
{functions_defs}

For the following question:
{prompts[4]['prompt']}

Determine which function should be called and what parameters should be passed to it.

Return ONLY a valid JSON object using EXACTLY this format:
{{"prompt": "<the original question>", "name": "<exact function name>", "parameters": {{"<parameter_name>": <parameter_value>}}}}

Rules:
- "prompt" must contain the original question exactly.
- "name" must be the exact name of one of the provided functions.
- "parameters" must contain the function arguments using the exact parameter names from its definition.
- Infer the parameter values from the question.
- Do not calculate or return the result of the function.
- Do not include any explanation or additional text.
- Do not use Markdown or code fences.
- Return exactly one JSON object nothing else.

Example:

Question:
What is the sum of 2 and 3?

Output:
{{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {{"a": 2.0, "b": 3.0}}}}
"""

input_ids = tokenizer.encode(prompt_text)

# Generate tokens one at a time
generated = input_ids

while True:
    try:
        logits = model.get_logits_from_input_ids(generated)

        next_token = max(range(len(logits)), key=lambda i: logits[i])
        if next_token == 151645:
            raise KeyboardInterrupt
        generated.append(next_token)
        response = tokenizer.decode(next_token)
        print(response, end="", flush=True)
    except KeyboardInterrupt:
        try:
            print()
            print()
            prompt = input("prompt: ")
            input_ids = tokenizer.encode(prompt)
            generated = input_ids
        except KeyboardInterrupt:
            exit()
