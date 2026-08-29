from typing import Set

from .selectable_tokenizer import SelectableTokenizer


class StringValidator:
    """Validates tokens for string parameter generation.

    Ensures that the generated string properly terminates with a quote
    and contains valid string characters.
    """

    def __init__(self, tokenizer: SelectableTokenizer) -> None:
        """Initializes the StringValidator.

        Args:
            tokenizer (Tokenizer): The tokenizer instance.
        """
        self._tokenizer = tokenizer
        self._end_ids: Set[int] = set()
        self._content_ids: Set[int] = set()
        for token, token_id in tokenizer.vocab.items():
            if '"' in token:
                self._end_ids.add(token_id)
            elif all(c not in token for c in ('"', "{", "}", ",")):
                self._content_ids.add(token_id)

    def is_valid_next(self, token_id: int) -> bool:
        """Checks if a token is a valid continuation of the string.

        Args:
            token_id (int): The ID of the token to check.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        return token_id in self._content_ids or token_id in self._end_ids

    def is_end(self, token_id: int) -> bool:
        """Checks if a token signifies the end of the string.

        Args:
            token_id (int): The ID of the token to check.

        Returns:
            bool: True if the token is an end token, False otherwise.
        """
        return token_id in self._end_ids
