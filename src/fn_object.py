from pydantic.dataclasses import dataclass as pydataclass
from typing import Dict, Any


@pydataclass
class FN_OBJECT:

    name: str
    description: str
    parameters: Dict[Any, Any]
    return_type: str

    def __hash__(self) -> int:
        return hash(self.name)
