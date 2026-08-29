from typing import Set, List, Any
import json

from .fn_object import FN_OBJECT
from .prompt_object import PROMPT_OBJECT


class JsonParser:
    """Parses and validates JSON input files for the project."""

    @staticmethod
    def parse_functions(filename: str) -> Set[FN_OBJECT]:
        """Parses the functions definition JSON file.

        Args:
            filename (str): The path to the functions JSON file.

        Returns:
            Set[FN_OBJECT]: A set of parsed function objects.

        Raises:
            ValueError: If the JSON is invalid, missing required fields, or has duplicate function names.
        """
        data = JsonParser._load_json(filename)

        if not isinstance(data, list):
            got = type(data).__name__
            raise ValueError(f"{filename}: expected a JSON array, got {got}")

        functions: Set[FN_OBJECT] = set()
        for i, entry in enumerate(data):
            got = type(entry).__name__
            if not isinstance(entry, dict):
                raise ValueError(f"{filename}[{i}]: expected object, got {got}")

            for field in ("name", "description"):
                if field not in entry:
                    raise ValueError(f"{filename}[{i}]: missing field '{field}'")
                if not isinstance(entry[field], str):
                    raise ValueError(f"{filename}[{i}]: '{field}' must be a string")

            params = entry.get("parameters")
            if not isinstance(params, dict):
                raise ValueError(
                    f"{filename}[{i}]: missing or " f"invalid field 'parameters'"
                )

            returns = entry.get("returns")
            if not isinstance(returns, dict):
                raise ValueError(
                    f"{filename}[{i}]: missing or " f"invalid field 'returns.type'"
                )
            elif "type" not in returns or not isinstance(returns.get("type"), str):
                raise ValueError(
                    f"{filename}[{i}]: missing or " f"invalid field 'returns.type'"
                )

            obj = FN_OBJECT(
                name=entry["name"],
                description=entry["description"],
                parameters=params,
                return_type=returns["type"],
            )

            if obj in functions:
                raise ValueError(
                    f"{filename}[{i}]: duplicate " f"function name '{obj.name}'"
                )
            functions.add(obj)

        return functions

    @staticmethod
    def parse_prompts(filename: str) -> List[PROMPT_OBJECT]:
        """Parses the prompts JSON file.

        Args:
            filename (str): The path to the prompts JSON file.

        Returns:
            List[PROMPT_OBJECT]: A list of parsed prompt objects.

        Raises:
            ValueError: If the JSON is invalid or missing the prompt field.
        """
        data = JsonParser._load_json(filename)

        if not isinstance(data, list):
            got = type(data).__name__
            raise ValueError(f"{filename}: expected a JSON array, got {got}")

        prompts: List[PROMPT_OBJECT] = []
        for i, entry in enumerate(data):
            got = type(entry).__name__
            if not isinstance(entry, dict):
                raise ValueError(f"{filename}[{i}]: expected object, got {got}")

            if "prompt" not in entry:
                raise ValueError(f"{filename}[{i}]: missing field 'prompt'")
            if not isinstance(entry["prompt"], str):
                raise ValueError(f"{filename}[{i}]: 'prompt' must be a string")

            prompts.append(PROMPT_OBJECT(prompt=entry["prompt"]))

        return prompts

    @staticmethod
    def _load_json(filename: str) -> Any:
        """Loads raw data from a JSON file.

        Args:
            filename (str): The path to the JSON file.

        Returns:
            Any: The parsed JSON data.

        Raises:
            ValueError: If the file is not found or has invalid syntax.
        """
        try:
            with open(filename) as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"{filename}: file not found")
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{filename}: invalid JSON syntax: {e.msg} "
                f"(line {e.lineno}, col {e.colno})"
            )
