import json
from typing import Set, List, Dict, Any

from .fn_object import FN_OBJECT


class PromptBuilder:
    """Builds the system prompt and few-shot examples for the LLM."""

    def __init__(self, functions: Set[FN_OBJECT]) -> None:
        """Initializes the PromptBuilder with available functions.

        Args:
            functions (Set[FN_OBJECT]): A set of available function definitions.
        """
        defs: List[Dict[str, Any]] = []
        for fn in functions:
            defs.append(
                {
                    "name": fn.name,
                    "description": fn.description,
                    "parameters": fn.parameters,
                    "returns": {"type": fn.return_type},
                }
            )
        self._function_defs_json = json.dumps(defs)

    def build(self, question: str) -> str:
        """Constructs the full prompt containing definitions and the question.

        Args:
            question (str): The natural language question to process.

        Returns:
            str: The fully formatted prompt.
        """
        return (
            "You are a function-calling model.\n\n"
            "Here are the available function definitions:\n"
            f"{self._function_defs_json}\n\n"
            "Example:\n"
            "Question: Replace all digits in 'a1b2' with asterisks\n"
            'Call: fn_substitute_string_with_regex(source_string="a1b2", '
            'regex="[0-9]+", replacement="*")\n\n'
            "For the following question:\n"
            f"{question}\n\n"
            "Determine which function should be called "
            "and what parameters should be passed to it."
        )
