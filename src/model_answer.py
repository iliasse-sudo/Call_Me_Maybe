from pydantic.dataclasses import dataclass as pydataclass
from typing import List, Any


@pydataclass
class ModelAnswer:
    """Represents the decoded function call from the model.

    Attributes:
        prompt (str): The original natural language prompt.
        name (str): The predicted function name to call.
        parameters (List[Any]): The ordered list of predicted parameter values.
    """

    prompt: str
    name: str
    parameters: List[Any]
