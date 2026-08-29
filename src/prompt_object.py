from pydantic.dataclasses import dataclass as pydataclass


@pydataclass
class PROMPT_OBJECT:
    """Represents an input prompt for function calling.

    Attributes:
        prompt (str): The natural language request.
    """

    prompt: str

    def __hash__(self) -> int:
        """Returns the hash of the prompt string for set operations."""
        return hash(self.prompt)
