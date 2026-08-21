import unicodedata
from typing import Dict, Literal, Set

class Normalizer():

    def __init__(self, normalizer_data: Dict[str, str]) -> None:
        form = normalizer_data.get("type", "NFC")
        if form not in self._accepted:
            raise RuntimeError("Unsuported normalizer type.")
        self.__type: Literal['NFC', 'NFD', 'NFKC', 'NFKD'] = form

    @property
    def _accepted(self) -> Set[Literal['NFC', 'NFD', 'NFKC', 'NFKD']]:
        return {'NFC', 'NFD', 'NFKC', 'NFKD'}

    def normalize(self, text: str) -> str:
        self.__type = "NFC" if not self.__type else self.__type
        return unicodedata.normalize(self.__type, text)
