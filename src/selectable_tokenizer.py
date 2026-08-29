from typing import Any, List

from llm_sdk import Small_LLM_Model
from tokenizer import Tokenizer


class SelectableTokenizer:
    """A wrapper that allows selecting between custom and model tokenizers.

    Provides unified encode and decode methods, delegating to either the
    custom tokenizer or the Hugging Face model tokenizer based on configuration.
    """

    use_model_encode: bool
    use_model_decode: bool

    def __init__(
        self,
        my_tokenizer: Tokenizer,
        model: Small_LLM_Model,
        use_model_encode: bool = False,
        use_model_decode: bool = False,
    ) -> None:
        """Initializes the SelectableTokenizer.

        Args:
            my_tokenizer (Tokenizer): The custom BPE tokenizer instance.
            model (Small_LLM_Model): The LLM SDK model instance containing the HF tokenizer.
            use_model_encode (bool, optional): If True, uses the model's encode method. Defaults to False.
            use_model_decode (bool, optional): If True, uses the model's decode method. Defaults to False.
        """
        self._my_tokenizer = my_tokenizer
        self._model = model
        self.use_model_encode = use_model_encode
        self.use_model_decode = use_model_decode

    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            List[int]: A list of integer token IDs.
        """
        if self.use_model_encode:
            encoded = self._model.encode(text)
            return [int(t) for t in encoded.tolist()[0]]
        return self._my_tokenizer.encode(text)

    def decode(self, ids: List[int] | int) -> str:
        """Decodes token IDs back into a string.

        Args:
            ids (List[int] | int): A single token ID or a list of token IDs.

        Returns:
            str: The decoded string.
        """
        if not self.use_model_decode:
            return self._my_tokenizer.decode(ids)
        if isinstance(ids, int):
            ids = [ids]
        decoded: str = self._model.decode(ids)
        return decoded

    @property
    def vocab(self) -> Any:
        """Retrieves the vocabulary mapping.

        Returns:
            Any: The vocabulary dictionary mapping tokens to IDs.
        """
        return self._my_tokenizer.vocab
