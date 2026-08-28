import json
from typing import Set, List, Dict, Any

from .fn_object import FN_OBJECT


class PromptBuilder:

    def __init__(self, functions: Set[FN_OBJECT]) -> None:
        defs: List[Dict[str, Any]] = []
        for fn in functions:
            defs.append({
                "name": fn.name,
                "description": fn.description,
                "parameters": fn.parameters,
                "returns": {"type": fn.return_type},
            })
        self._function_defs_json = json.dumps(defs)

    def build(self, question: str) -> str:
        return (
            "You are a function-calling model.\n\n"
            "Here are the available function definitions:\n"
            f"{self._function_defs_json}\n\n"
            "For the following question:\n"
            f"{question}\n\n"
            "Determine which function should be called "
            "and what parameters should be passed to it."
        )
