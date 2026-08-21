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
prompt = input("prompt: ")
normalized_text = tokenizer._normalize(prompt)
pre_tokenized_text = tokenizer._pre_tokenize(normalized_text)
t = perf_counter()
encoded = tokenizer.encode(prompt)
my_encode_time = perf_counter() - t
t = perf_counter()
model_encoded = model.encode(prompt)[0].tolist()
model_encode_time = perf_counter() - t
t = perf_counter()
decoded = tokenizer.decode(encoded)
my_decode_time = perf_counter() - t
t = perf_counter()
model_decoded = model.decode(model_encoded)
model_decode_time = perf_counter() - t

print(f"original text:      {repr(prompt)}")
print(f"normalized text:    {repr(normalized_text)}")
print(f"pre_tokenized_text: {pre_tokenized_text}")
print(f"my encoder:         {encoded} {my_encode_time:.5f}")
print(f"model encoder:      {model_encoded} {model_encode_time:.5f}")
print(f"my decoder:         {repr(decoded)} {my_decode_time:.5f}")
print(f"model decoder:      {repr(model_decoded)} {model_decode_time:.5f}")

exit()


input_ids = model.encode(prompt)

# Generate tokens one at a time
generated = input_ids[0].tolist()

while True:
    try:
        logits = model.get_logits_from_input_ids(generated)

        next_token = max(range(len(logits)), key=lambda i: logits[i])
        if next_token == 151645:
            raise KeyboardInterrupt
        generated.append(next_token)
        response = model.decode(next_token)
        print(response, end="", flush=True)
    except KeyboardInterrupt:
        try:
            print()
            print()
            prompt = input("prompt: ")
            input_ids = model.encode(prompt)
            generated = input_ids[0].tolist()
        except KeyboardInterrupt:
            exit()
