from typing import List, Set, Dict, Any

from .json_parser import JsonParser
from .selectable_tokenizer import SelectableTokenizer


class FnNameValidator:
    """Validates tokens for function name generation using a Trie.

    Ensures that generated function names exactly match one of the
    available function definitions.
    """

    functions: Set[Any]
    _trie: Dict[Any, Any]

    def __init__(
        self, definitions_path: str, tokenizer: SelectableTokenizer
    ) -> None:
        """Initializes the FnNameValidator and builds the Trie.

        Args:
            definitions_path (str): Path to the functions JSON file.
            tokenizer (Tokenizer): The tokenizer instance.
        """
        self.functions = JsonParser.parse_functions(definitions_path)
        self.tokenizer = tokenizer

        self._trie = {}
        for fn in self.functions:
            token_ids = self.tokenizer.encode(fn.name)
            node = self._trie
            for tid in token_ids:
                if tid not in node:
                    node[tid] = {}
                node = node[tid]
            node["__end__"] = True

    def get_valid_next_tokens(self, token_ids: List[int]) -> List[int]:
        """Retrieves the list of valid next tokens based on current context.

        Args:
            token_ids (List[int]): The currently generated function name
                tokens.

        Returns:
            List[int]: A list of allowed next token IDs, optionally including
                the quote token.
        """
        node = self._trie
        for tid in token_ids:
            if tid not in node:
                return []
            node = node[tid]
        return [k for k in node if k != "__end__"] + (
            self.tokenizer.encode('"') if "__end__" in node else []
        )
