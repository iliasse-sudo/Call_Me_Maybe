*This project has been created as part of the 42 curriculum by ibaya.*

# Call Me Maybe 🤙

## Description
**Call Me Maybe** is an introduction to function calling in Large Language Models (LLMs). The project demonstrates how to make a small, 0.6B parameter model (Qwen/Qwen3-0.6B) speak the language of computers by reliably translating natural language requests into structured, machine-executable function calls.

Instead of relying purely on prompting and hoping the LLM outputs valid JSON, this project implements **constrained decoding**. This technique guides the model token-by-token, guaranteeing 100% valid JSON output that strictly adheres to a predefined schema. 

## Instructions

### Prerequisites
- Python 3.10 or later
- `uv` package manager

### Installation
Clone the repository and install dependencies using the provided Makefile:
```bash
make install
```
*(This runs `uv sync` to set up the virtual environment and install required packages like `torch`, `transformers`, `numpy`, and `pydantic`).*

### Execution
Run the program with default arguments (processing `data/input/function_calling_tests.json`):
```bash
make run
```

Or run it manually with custom paths:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Other Commands
- **Linting**: `make lint` or `make lint-strict` to run `flake8` and `mypy`.
- **Debug**: `make debug` to run with `pdb`.
- **Clean**: `make clean` to remove cache and compiled files.

---

## Algorithm Explanation
The core mechanism used to ensure 100% valid JSON is **constrained decoding**. Here is how it works step-by-step:

1. **Structural Framing:** Instead of asking the model to generate the entire JSON object, the system hardcodes the structural parts. It writes `{"prompt": "...", "name": "` directly into the context.
2. **Logit Masking:** When it is time for the model to predict the function name or parameter values, the decoder intercepts the probability distribution (logits) of the next token.
3. **Validation Filtering:** The system identifies which tokens in the vocabulary are valid next steps based on the expected type (e.g., matching a function name in a Trie, or checking if a token contains valid digits for a number).
4. **Enforcing Constraints:** Invalid tokens have their logits set to negative infinity (`-inf`).
5. **Selection:** The model is forced to choose the highest probability token from the *remaining valid tokens*, guaranteeing that the output will always parse correctly and match the schema.

## Design Decisions
- **Trie-based Function Name Validation:** Function names are encoded into a Trie (prefix tree) of token IDs. This ensures the model can only generate names that exist in `functions_definition.json`.
- **Stateful Token Validators:** Individual classes (`StringValidator`, `NumberValidator`, `IntegerValidator`, `BooleanValidator`) handle the rules for parameter generation, terminating when logical endpoints (like `,` or `}`) are reached.
- **Pydantic Data Models:** `FN_OBJECT`, `PROMPT_OBJECT`, and `ModelAnswer` use Pydantic for robust runtime type checking and structured data handling.
- **Bonus Implementation:** The project implements its own BPE Tokenizer (`tokenizer/`) rather than relying exclusively on the Hugging Face implementations for encoding and decoding.

## Performance Analysis
- **Accuracy:** The system achieves near-perfect structural accuracy because syntactic errors are mathematically impossible due to the `-inf` logit masking. Semantic accuracy relies on the 0.6B model's reasoning capabilities, which are greatly aided by the structured system prompt.
- **Speed:** By using a lightweight 500 million parameter model, inference runs quickly on standard hardware.
- **Reliability:** 100% valid JSON generation is guaranteed. There are no trailing commas, missing brackets, or hallucinated extra keys.

## Challenges Faced
- **Token Boundaries:** Models often split words across tokens in unpredictable ways (e.g., spaces attached to the beginning of words). Building robust validators (especially for strings and numbers) that account for how the BPE tokenizer chunks characters was challenging.
- **Large Vocabulary Iteration:** Evaluating validity across a vocabulary of ~150,000 tokens can be slow. This was solved by pre-computing valid token sets where possible (e.g., caching allowed digit tokens in `NumberValidator`).
- **State Management:** Tracking the state of boolean generation token-by-token required a careful state-machine implementation to ensure sequences like `"true"` and `"false"` were correctly forced.

## Testing Strategy
The implementation was validated using the provided test files in `data/input/`. 
1. **Unit Testing:** Individual validators were checked against edge cases (e.g., floating point numbers, negative numbers, strings with quotes).
2. **End-to-End Testing:** The main pipeline was run against `function_calling_tests.json` and the output was compared against `function_calling_corrections.json` to ensure both structural validity and semantic correctness.
3. **Error Handling:** Tested against missing files and malformed input JSONs, ensuring the program exits gracefully and logs clear error messages to `stderr`.

## Example Usage

**Input Question:**
> "What is the sum of 265 and 345?"

**Command:**
```bash
uv run python -m src
```

**Console Output during generation:**
```
[2/11] What is the sum of 265 and 345?
{"prompt": "What is the sum of 265 and 345?", "name": "fn_add_numbers", "parameters": {"a": 265.0, "b": 345.0}}
```

**Generated JSON (`data/output/function_calling_results.json`):**
```json
{
  "prompt": "What is the sum of 265 and 345?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 265.0,
    "b": 345.0
  }
}
```

## Resources
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- **AI Usage:** AI was used during this project to analyze the architecture of the constrained decoding pipeline, identify state mutation bugs in token validators, and assist in generating the structured documentation for this README.
