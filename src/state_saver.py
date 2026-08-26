from pydantic.dataclasses import dataclass as pydataclass
from typing import Dict, Any


@pydataclass
class State_Saver:

    cur_prompt: str
    prompt: str
    function_name: str
    parameters: Dict[Any, Any]
