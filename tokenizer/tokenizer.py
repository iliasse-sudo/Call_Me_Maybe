import json
from typing import List, Dict, Any
from pathlib import Path
from .normalizer import Normalizer
from .pre_tokenizer import Pre_tokenizer
from .encoder import Encoder
from .decoder import Decoder

__all__ = ["Tokenizer", "Normalizer", "Pre_tokenizer", "Encoder", "Decoder"]

class Tokenizer():

    init = False

    def __init__(self, tokenizer_file: Path):
        with open(tokenizer_file, encoding="utf-8") as f:
            tokenizer_data: Dict[str, Any] = json.load(f)
            self.added_tokens: List[Dict[str, Any]] =\
                  tokenizer_data.get("added_tokens", [])
            self.normalizer_data: Dict[str, str]=\
                  tokenizer_data.get("normalizer", {})
            self.pre_tokenizer_data = tokenizer_data.get("pre_tokenizer", {})
            self.post_processor_data = tokenizer_data.get("post_processor", {})
            self.decoder_data = tokenizer_data.get("decoder", {})
            self.model_data: Dict[str, Any] = tokenizer_data.get("model", {})
            self.vocab = self.model_data.get("vocab", {})
            self.merges = self.model_data.get("merges", [])
        self._normalizer = Normalizer(self.normalizer_data)
        self._pretokenizer = Pre_tokenizer(self.pre_tokenizer_data,
                                           self._build_mapping)
        self._encoder = Encoder(self.model_data, self._normalize,
                                self._pre_tokenize)
        self._decoder = Decoder(self.decoder_data, self.vocab,
                                self._build_mapping)
        self.init = True

    @staticmethod
    def _build_mapping() -> Dict[int, str]:
        ranges = [("!", "~"), ("¡", "¬"), ("®", "ÿ")]
        mapping = {}
        normal_chars = set()
        for start, end in ranges:
            for el in range(ord(start), ord(end) + 1):
                normal_chars.add(el)
                mapping.update({el: chr(el)})

        n = 0
        for el in range(0, 256):
            if el not in normal_chars:
                mapping.update({el: chr(n + 256)})
                n += 1
        return mapping

    def _normalize(self, text: str) -> str:
        if self.init is False:
            raise RuntimeError("Please call the Tokenizer.__init__() first")
        return self._normalizer.normalize(text)

    def _pre_tokenize(self, text: str) -> List[str]:
        if self.init is False:
            raise RuntimeError("Please call the Tokenizer.__init__() first")
        return self._pretokenizer.pre_tokenize(text)

    def encode(self, text: str) -> List[int]:
        if self.init is False:
            raise RuntimeError("Please call the Tokenizer.__init__() first")
        return self._encoder.encode(text)

    def decode(self, ids: List[int] | int) -> str:
        if self.init is False:
            raise RuntimeError("Please call the Tokenizer.__init__() first")
        return self._decoder.decode(ids)
