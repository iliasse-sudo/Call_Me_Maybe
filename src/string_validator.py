from tokenizer import Tokenizer


class StringValidator:

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    def is_valid_next(self, token_id: int) -> bool:
        return not self.is_end(token_id)

    def is_end(self, token_id: int) -> bool:
        decoded: str = self._tokenizer.decode(token_id)
        return decoded.startswith('"')
