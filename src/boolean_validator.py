from tokenizer import Tokenizer


class BooleanValidator:

    def __init__(self, tokenizer: Tokenizer) -> None:
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
        self._state = 0

    def _advance(self, ch: str) -> int:
        return self._TRANSITIONS.get(self._state, {}).get(ch, -1)

    def is_valid_next(self, token_id: int) -> bool:
        decoded: str = self._tokenizer.decode(token_id)
        saved = self._state
        for ch in decoded:
            self._state = self._advance(ch)
            if self._state == -1:
                self._state = saved
                return False
        return True

    def is_end(self, token_id: int) -> bool:
        _ = token_id
        return self._state in self._END_STATES
