from typing import Dict, Any, Callable, List

class Decoder():

    def __init__(self, decoder_data: Dict[str, Any],
                 vocab: Dict[str, int],
                 mapping_builder: Callable[[], Dict[int, str]]) -> None:
        if decoder_data.get("type", "") != "ByteLevel":
            raise RuntimeError("Unsuported decoder type.")
        self._decoder_data = decoder_data
        self._vocab = vocab
        self._build_mapping = mapping_builder
        self.__reversed_mapping = self._reverse_mapping(self._build_mapping())
        self.__reversed_vocab = self._reverse_vocab(self._vocab)

    def _reverse_mapping(self, mapping: Dict[int, str]) -> Dict[str, int]:
        reversed_mapping: Dict[str, int] = {}
        for key, value in mapping.items():
            reversed_mapping.update({value: key})
        return reversed_mapping

    def _reverse_vocab(self, vocab: Dict[str, int]) -> Dict[int, str]:
        reversed_vocab: Dict[int, str] = {}
        for key, value in vocab.items():
            reversed_vocab.update({value: key})
        return reversed_vocab

    def _get_tokens_from_ids(self, ids: List[int]) -> List[str]:
        tokens = []
        for tok_id in ids:
            tok = self.__reversed_vocab.get(tok_id)
            if tok is None:
                raise RuntimeError("Couldn't get token for token_id: "
                                   f"{tok_id}")
            tokens.append(tok)
        return tokens

    def _bytelevel(self, tokens: List[str]) -> str:
        res = []
        for token in tokens:
            for b in token:
                org_byte = self.__reversed_mapping.get(b)
                if org_byte is None:
                    raise RuntimeError("Couldn't decode character "
                                       f"'{org_byte}'")
                res.append(org_byte)
        return bytes(res).decode("utf-8")

    def decode(self, ids: List[int] | int) -> str:
        if isinstance(ids, int):
            ids = [ids]
        tokens = self._get_tokens_from_ids(ids)
        text = self._bytelevel(tokens)
        return text
