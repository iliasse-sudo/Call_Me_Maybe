from pydantic.dataclasses import dataclass as pydataclass


@pydataclass
class PROMPT_OBJECT:

    prompt: str

    def __hash__(self) -> int:
        return hash(self.prompt)
