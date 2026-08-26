from typing import List, Set, Dict, Any

from .json_parser import JsonParser
from tokenizer import Tokenizer


class FnNameValidator:

    functions: Set[Any]
    _trie: Dict[Any, Any]

    def __init__(self, definitions_path: str, tokenizer: Tokenizer) -> None:
        self.functions = JsonParser.parse_functions(definitions_path)

        self._trie = {}
        for fn in self.functions:
            token_ids = tokenizer.encode(fn.name)
            node = self._trie
            for tid in token_ids:
                if tid not in node:
                    node[tid] = {}
                node = node[tid]
            node["__end__"] = True

    def get_valid_next_tokens(self, token_ids: List[int]) -> List[int]:
        node = self._trie
        for tid in token_ids:
            if tid not in node:
                return []
            node = node[tid]
        return [k for k in node if k != "__end__"] + (
            ['"'] if "__end__" in node else []
        )
