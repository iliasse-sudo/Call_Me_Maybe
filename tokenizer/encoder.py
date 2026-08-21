from typing import Dict, Any, List, Set, Callable, Tuple
from itertools import pairwise

class Encoder():

    init = False

    def __init__(self, model_data: Dict[str, Any],
                 normalizer: Callable[[str], str],
                 pre_tokenizer: Callable[[str], List[str]]) -> None:

        self._model_data = model_data
        self._type = self._model_data.get("type", "BPE")
        self._vocab: Dict[str, int] = self._model_data.get("vocab", {})
        self._merges: List[List[str]]= self._model_data.get("merges", [])
        self._normalize = normalizer
        self._pretokenize = pre_tokenizer
        self.init = True

    def _get_pairs(self, to_tokenize: List[List[str]]
                   ) -> List[Set[Tuple[str, str]]]:
        paired: List[Set[Tuple[str, str]]]= []
        for pre_tok in to_tokenize:
            paired.append(self._get_set_pairs(pre_tok))
        return paired

    def _get_set_pairs(self, pre_tok: List[str]) -> Set[Tuple[str, str]]:
        tok_set = set()
        for pair in pairwise(pre_tok):
            tok_set.add(pair)
        return tok_set

    def _to_chars(self, to_split: List[str]) -> List[List[str]]:
        res: List[List[str]] = []
        for pre_tok in to_split:
            res.append(list(pre_tok))
        return res

    def bpe(self, to_tokenize: List[str]) -> List[List[str]]:
        to_merge = self._to_chars(to_tokenize)
        pairs = self._get_pairs(to_merge)
        merges_lenght = len(self._merges)
        i = 0
        while i < merges_lenght:
            restart = False
            for idx, pairs_set in enumerate(pairs, 0):
                if tuple(self._merges[i]) in pairs_set:
                    n = 0
                    pre_tok = to_merge[idx]
                    pre_tok_lenght = len(pre_tok)
                    while n + 1 < pre_tok_lenght:
                        if [pre_tok[n], pre_tok[n + 1]] == self._merges[i]:
                            pre_tok[n] += pre_tok[n + 1]
                            pre_tok.pop(n + 1)
                            pairs[idx] = self._get_set_pairs(to_merge[idx])
                            restart = True
                            break
                        n += 1
            i = 0 if restart is True else i + 1
        return to_merge

    def encode(self, text: str) -> List[int]:
        normalized = self._normalize(text)
        pre_tokenized = self._pretokenize(normalized)
        final_merge = []
        for pre_tok in self.bpe(pre_tokenized):
            for tok in pre_tok:
                final_merge.append(tok)
        ids: List[int] = []
        for tok in final_merge:
            tok_id = self._vocab.get(tok)
            if tok_id is None:
                raise RuntimeError("Couldn't get token_id for token: "
                                   f"{tok}")
            ids.append(tok_id)
        return ids
