"""Entry point of the call-me-maybe function-calling pipeline.

Loads the Qwen3-0.6B model, runs constrained decoding over the list
of prompts, and writes the resulting function calls to a JSON file.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from llm_sdk import Small_LLM_Model
from tokenizer import Tokenizer

from .constrained_decoder import ConstrainedDecoder
from .fn_object import FN_OBJECT
from .json_parser import JsonParser
from .model_answer import ModelAnswer
from .prompt_builder import PromptBuilder
from .selectable_tokenizer import SelectableTokenizer

DEFAULT_FUNCTIONS_DEF_FILE = "data/input/functions_definition.json"
DEFAULT_INPUT_FILE = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_FILE = "data/output/function_calling_results.json"


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments for the program."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function "
        "calls using a constrained LLM decoder.",
    )
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS_DEF_FILE,
        help="Path to the JSON file describing available functions "
        f"(default: {DEFAULT_FUNCTIONS_DEF_FILE}).",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help="Path to the JSON file containing the prompts "
        f"(default: {DEFAULT_INPUT_FILE}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Path to the JSON file where results are written "
        f"(default: {DEFAULT_OUTPUT_FILE}).",
    )
    return parser.parse_args(argv)


def build_answer_json(
    answer: ModelAnswer,
    functions_by_name: Dict[str, FN_OBJECT],
) -> Dict[str, object]:
    """Rebuild the JSON representation of a generated answer."""
    function = functions_by_name[answer.name]
    parameters = {
        key: value for key, value in zip(function.parameters, answer.parameters)
    }
    return {
        "prompt": answer.prompt,
        "name": answer.name,
        "parameters": parameters,
    }


def main() -> None:
    """Run the full pipeline and write the results to the output file."""
    args = parse_args(sys.argv[1:])

    try:
        model = Small_LLM_Model()

        tokenizer_file = Path(model.get_path_to_tokenizer_file())
        tokenizer = SelectableTokenizer(Tokenizer(tokenizer_file), model, True)

        decoder = ConstrainedDecoder(
            tokenizer,
            args.functions_definition,
        )

        functions = JsonParser.parse_functions(args.functions_definition)
        functions_by_name = {f.name: f for f in functions}

        prompt_builder = PromptBuilder(functions)
        prompts = JsonParser.parse_prompts(args.input)
    except Exception as e:
        print(f"Error: initialization failed: {e}", file=sys.stderr)
        sys.exit(1)

    results: List[Dict[str, object]] = []
    for i, prompt_obj in enumerate(prompts, start=1):
        print(f"\n[{i}/{len(prompts)}] {prompt_obj.prompt}")
        try:
            answer = decoder.decode(
                prompt_builder,
                prompt_obj,
                model.get_logits_from_input_ids,
            )
        except Exception as e:
            print(
                f"Error: generation failed for '{prompt_obj.prompt}': " f"{e}",
                file=sys.stderr,
            )
            continue
        print()
        results.append(build_answer_json(answer, functions_by_name))

    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
    except OSError as exc:
        print(f"Error: could not write output file: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nSuccessfully wrote {len(results)} result(s) to " f"{args.output}")


if __name__ == "__main__":
    main()
