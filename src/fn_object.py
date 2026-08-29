from pydantic.dataclasses import dataclass as pydataclass
from typing import Dict, Any


@pydataclass
class FN_OBJECT:
    """Represents a function definition.

    Attributes:
        name (str): The name of the function.
        description (str): A description of what the function does.
        parameters (Dict[Any, Any]): The expected parameters and their types.
        return_type (str): The type of the return value.
    """

    name: str
    description: str
    parameters: Dict[Any, Any]
    return_type: str

    def __hash__(self) -> int:
        """Returns the hash of the function name for set operations."""
        return hash(self.name)
