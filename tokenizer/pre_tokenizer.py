from typing import Dict, Any, List, Callable

class Pre_tokenizer():

    def __init__(self, pre_tokenizer_data: Dict[str, Any],
                 mapping_builder: Callable[[], Dict[int, str]]) -> None:
        self.pre_tokenizers: List[Dict[str, Any]] = []
        _type: str = pre_tokenizer_data.get("type", "")
        if _type == "Sequence":
            for pre_tokenizer in pre_tokenizer_data.get("pretokenizers", {}):
                self.pre_tokenizers.append(pre_tokenizer)

        else:
            self.pre_tokenizers.append(pre_tokenizer_data)

        self._build_mapping = mapping_builder
        self._mapping = self._build_mapping()

    def _split(self, data: Dict[str, Any], text: str) -> List[str]:
        import regex
        regex_pattern: str | None = data.get("pattern", {}).get("Regex", None)
        if regex_pattern is None:
            raise RuntimeError("No regex pattern found")
        matches = regex.findall(regex_pattern, text)
        return matches

    def _byte_level(self, inputs: List[str])-> List[str]:
        self._mapping = self._build_mapping()

        returns = []
        for el in inputs:
            new = ""
            for b in el.encode('utf-8'):
                new += self._mapping[b]
            returns.append(new)
        return returns

    def pre_tokenize(self, text: str) -> List[str]:
        for tokenizer_data in self.pre_tokenizers:
            _type = tokenizer_data.get("type", None)
            if _type == "Split":
                split_result = self._split(tokenizer_data, text)
            elif _type == "ByteLevel":
                result = self._byte_level(split_result)
            else:
                raise RuntimeError("Unsuported pre_tokenizer type")
        return result
