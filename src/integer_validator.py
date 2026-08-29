from typing import Set, List
from .selectable_tokenizer import SelectableTokenizer


class IntegerValidator:
    """Validates tokens for integer parameter generation.

    Ensures that generated tokens form a valid integer string.
    """

    allowed_tokens: Set[int]
    end_tokens: Set[int]

    def __init__(self, tokenizer: SelectableTokenizer) -> None:
        """Initializes the IntegerValidator.

        Args:
            tokenizer (Tokenizer): The tokenizer instance.
        """
        self._tokenizer = tokenizer
        self.allowed_tokens = set()
        self.end_tokens = set()
        for ch in "0123456789-+":
            token_ids = tokenizer.encode(ch)
            self.allowed_tokens.update(token_ids)
        for ch in ",}":
            token_ids = tokenizer.encode(ch)
            self.end_tokens.update(token_ids)

    def _has_digit(self, token_ids: List[int]) -> bool:
        """Checks if the sequence of tokens contains at least one digit.

        Args:
            token_ids (List[int]): The list of generated token IDs.

        Returns:
            bool: True if a digit is present, False otherwise.
        """
        if not token_ids:
            return False
        decoded = self._tokenizer.decode(token_ids)
        return any(ch.isdigit() for ch in decoded)

    def get_valid_next_tokens(self, token_ids: List[int]) -> List[int]:
        """Retrieves the list of valid next tokens based on current context.

        Args:
            token_ids (List[int]): The currently generated tokens.

        Returns:
            List[int]: A list of allowed next token IDs.
        """
        allowed = list(self.allowed_tokens)
        if self._has_digit(token_ids):
            allowed.extend(self.end_tokens)
        return allowed

    def is_valid_next(self, token_id: int) -> bool:
        """Checks if a single token is valid (fallback).

        Args:
            token_id (int): The candidate token ID.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        return token_id in self.allowed_tokens

    def is_end(self, token_id: int) -> bool:
        """Checks if a token signifies the end of the integer value.

        Args:
            token_id (int): The token ID to check.

        Returns:
            bool: True if it's an end token, False otherwise.
        """
        decoded: str = self._tokenizer.decode(token_id)
        return decoded.endswith(",") or decoded.endswith("}")
