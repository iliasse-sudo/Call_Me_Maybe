from typing import Set
from tokenizer import Tokenizer


class NumberValidator:

    allowed_tokens: Set[int]

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self.allowed_tokens = set()
        for ch in "0123456789-+.":
            token_ids = tokenizer.encode(ch)
            self.allowed_tokens.update(token_ids)

    def is_valid_next(self, token_id: int) -> bool:
        return token_id in self.allowed_tokens

    def is_end(self, token_id: int) -> bool:
        decoded: str = self._tokenizer.decode(token_id)
        return decoded.startswith(",")
