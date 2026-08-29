from typing import List, Dict, Any, Callable
from .selectable_tokenizer import SelectableTokenizer
from .fn_name_validator import FnNameValidator
from .fn_object import FN_OBJECT
from .number_validator import NumberValidator
from .integer_validator import IntegerValidator
from .string_validator import StringValidator
from .boolean_validator import BooleanValidator
from .prompt_builder import PromptBuilder
from .prompt_object import PROMPT_OBJECT
from .model_answer import ModelAnswer


class ConstrainedDecoder:
    """Core engine for constrained decoding of function calls.

    Uses token logit masking to ensure the language model strictly adheres
    to the predefined JSON schema and types.
    """

    _tokenizer: SelectableTokenizer
    _fn_name_validator: FnNameValidator
    _number_validator: NumberValidator
    _integer_validator: IntegerValidator
    _string_validator: StringValidator
    _boolean_validator: BooleanValidator
    _function_defs: Dict[str, FN_OBJECT]

    def __init__(self, tokenizer: SelectableTokenizer, definitions_path: str) -> None:
        """Initializes the constrained decoder.

        Args:
            tokenizer (SelectableTokenizer): The wrapper for tokenizer methods.
            definitions_path (str): Path to the function definitions JSON file.
        """
        self._tokenizer = tokenizer
        self._fn_name_validator = FnNameValidator(definitions_path, tokenizer)
        self._number_validator = NumberValidator(tokenizer)
        self._integer_validator = IntegerValidator(tokenizer)
        self._string_validator = StringValidator(tokenizer)
        self._boolean_validator = BooleanValidator(tokenizer)

        self._function_defs = {}
        for fn in self._fn_name_validator.functions:
            self._function_defs[fn.name] = fn

    def get_masked_logits(
        self, logits: List[float], allowed_tokens: List[int]
    ) -> List[float]:
        """Masks out disallowed tokens by setting their logits to negative infinity.

        Args:
            logits (List[float]): The original model logits.
            allowed_tokens (List[int]): The list of token IDs that are permitted.

        Returns:
            List[float]: The masked logits.
        """
        masked = [-float("inf")] * len(logits)
        for tid in allowed_tokens:
            if tid < len(masked):
                masked[tid] = logits[tid]
        return masked

    def write(self, text: str) -> List[int]:
        """Prints text to standard output and returns its token IDs.

        Args:
            text (str): The text to write and encode.

        Returns:
            List[int]: The token IDs corresponding to the text.
        """
        token_ids = self._tokenizer.encode(text)
        print(text, end="", flush=True)
        return token_ids

    def predict(
        self,
        logits: List[float],
        validator: Any,
        token_ids: List[int],
    ) -> int:
        """Selects the best valid token based on the provided validator and logits.

        Args:
            logits (List[float]): The raw logits from the model.
            validator (Any): The stateful validator instance to use.
            token_ids (List[int]): The currently generated sequence of token IDs.

        Returns:
            int: The ID of the highest probability valid token.
        """
        if hasattr(validator, "get_valid_next_tokens"):
            allowed = validator.get_valid_next_tokens(token_ids)
        else:
            allowed = [i for i in range(len(logits)) if validator.is_valid_next(i)]
        masked = self.get_masked_logits(logits, allowed)
        return max(range(len(masked)), key=lambda i: masked[i])

    def decode(
        self,
        prompt_builder: PromptBuilder,
        prompt_obj: PROMPT_OBJECT,
        logits_fn: Callable[[List[int]], List[float]],
    ) -> ModelAnswer:
        """Decodes a single prompt into a structured function call.

        Args:
            prompt_builder (PromptBuilder): The builder for formatting the prompt.
            prompt_obj (PROMPT_OBJECT): The target natural language prompt.
            logits_fn (Callable[[List[int]], List[float]]): Function yielding logits.

        Returns:
            ModelAnswer: The resulting decoded function name and parameters.
        """
        prompt_text = prompt_builder.build(prompt_obj.prompt)
        context: List[int] = self._tokenizer.encode(prompt_text)

        context.extend(self.write('{"prompt": '))
        context.extend(self.write(f'"{prompt_obj.prompt}"'))
        context.extend(self.write(', "name": "'))

        generated: List[int] = []
        while True:
            logits = logits_fn(context + generated)
            token_id = self.predict(logits, self._fn_name_validator, generated)
            generated.append(token_id)
            print(
                self._tokenizer.decode(token_id),
                end="",
                flush=True,
            )
            if not self._fn_name_validator.get_valid_next_tokens(generated):
                break

        context.extend(generated)

        fn_name = self._tokenizer.decode(generated)
        if fn_name.endswith('"'):
            fn_name = fn_name[:-1]
        fn_obj = self._function_defs[fn_name]

        context.extend(self.write(', "parameters": {'))

        params = list(fn_obj.parameters.items())
        param_values: List[Any] = []
        for i, (key, spec) in enumerate(params):
            context.extend(self.write(f'"{key}": '))

            param_type = spec.get("type", "string")
            validator: Any
            if param_type == "number":
                validator = self._number_validator
            elif param_type == "integer":
                validator = self._integer_validator
            elif param_type == "boolean":
                validator = self._boolean_validator
                validator.reset()
            else:
                context.extend(self.write('"'))
                validator = self._string_validator

            value_tokens: List[int] = []
            while True:
                logits = logits_fn(context + value_tokens)
                token_id = self.predict(logits, validator, value_tokens)
                if validator.is_end(token_id):
                    if param_type == "string":
                        context.extend(self.write('"'))
                    break
                value_tokens.append(token_id)
                print(
                    self._tokenizer.decode(token_id),
                    end="",
                    flush=True,
                )

            context.extend(value_tokens)

            value_str = self._tokenizer.decode(value_tokens)
            if param_type == "number":
                param_values.append(float(value_str))
            elif param_type == "integer":
                param_values.append(int(value_str))
            elif param_type == "boolean":
                param_values.append(value_str == "true")
            else:
                param_values.append(value_str)

            if i < len(params) - 1:
                context.extend(self.write(", "))

        self.write("}}")
        return ModelAnswer(
            prompt=prompt_obj.prompt,
            name=fn_name,
            parameters=param_values,
        )
