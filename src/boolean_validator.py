from .selectable_tokenizer import SelectableTokenizer


class BooleanValidator:
    """Validates tokens for boolean parameter generation.

    Uses a finite state machine to enforce the generation of
    either 'true' or 'false'.
    """

    def __init__(self, tokenizer: SelectableTokenizer) -> None:
        """Initializes the BooleanValidator.

        Args:
            tokenizer (Tokenizer): The tokenizer instance.
        """
        self._tokenizer = tokenizer
        self._state = 0
        self._TRANSITIONS = {
            0: {"t": 1, "f": 5},
            1: {"r": 2},
            2: {"u": 3},
            3: {"e": 4},
            5: {"a": 6},
            6: {"l": 7},
            7: {"s": 8},
            8: {"e": 9},
        }
        self._END_STATES = {4, 9}

    def reset(self) -> None:
        """Resets the state machine to the initial state."""
        self._state = 0

    def _advance(self, ch: str) -> int:
        """Advances the state machine given a character.

        Args:
            ch (str): The character to process.

        Returns:
            int: The new state, or -1 if invalid.
        """
        return self._TRANSITIONS.get(self._state, {}).get(ch, -1)

    def is_valid_next(self, token_id: int) -> bool:
        """Checks if a token is a valid continuation of a boolean string.

        Args:
            token_id (int): The candidate token ID.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        decoded: str = self._tokenizer.decode(token_id)
        saved = self._state
        for ch in decoded:
            self._state = self._advance(ch)
            if self._state == -1:
                self._state = saved
                return False
        return True

    def is_end(self, token_id: int) -> bool:
        """Checks if the generated tokens form a complete boolean.

        Args:
            token_id (int): The candidate token ID.

        Returns:
            bool: True if the state machine reached an end state,
                False otherwise.
        """
        _ = token_id
        return self._state in self._END_STATES
