from typing import List, Dict, Any, Callable
from tokenizer import Tokenizer
from .fn_name_validator import FnNameValidator
from .fn_object import FN_OBJECT
from .number_validator import NumberValidator
from .string_validator import StringValidator
from .boolean_validator import BooleanValidator
from .prompt_builder import PromptBuilder
from .prompt_object import PROMPT_OBJECT


class ConstrainedDecoder:

    _tokenizer: Tokenizer
    _fn_name_validator: FnNameValidator
    _number_validator: NumberValidator
    _string_validator: StringValidator
    _boolean_validator: BooleanValidator
    _function_defs: Dict[str, FN_OBJECT]

    def __init__(
        self, tokenizer: Tokenizer, definitions_path: str
    ) -> None:
        self._tokenizer = tokenizer
        self._fn_name_validator = FnNameValidator(
            definitions_path, tokenizer
        )
        self._number_validator = NumberValidator(tokenizer)
        self._string_validator = StringValidator(tokenizer)
        self._boolean_validator = BooleanValidator(tokenizer)

        self._function_defs = {}
        for fn in self._fn_name_validator.functions:
            self._function_defs[fn.name] = fn

    def get_masked_logits(
        self, logits: List[float], allowed_tokens: List[int]
    ) -> List[float]:
        masked = [-float('inf')] * len(logits)
        for tid in allowed_tokens:
            if tid < len(masked):
                masked[tid] = logits[tid]
        return masked

    def write(self, text: str) -> List[int]:
        token_ids = self._tokenizer.encode(text)
        decoded = self._tokenizer.decode(token_ids)
        print(decoded, end="", flush=True)
        return token_ids

    def predict(
        self,
        logits: List[float],
        validator: Any,
        token_ids: List[int],
    ) -> int:
        if hasattr(validator, 'get_valid_next_tokens'):
            allowed = validator.get_valid_next_tokens(token_ids)
        else:
            allowed = [
                i for i in range(len(logits))
                if validator.is_valid_next(i)
            ]
        masked = self.get_masked_logits(logits, allowed)
        return max(range(len(masked)), key=lambda i: masked[i])

    def decode(
        self,
        model: Any,
        prompt_builder: PromptBuilder,
        prompt_obj: PROMPT_OBJECT,
        logits_fn: Callable,
    ) -> str:
        prompt_text = prompt_builder.build(prompt_obj.prompt)
        context: List[int] = self._tokenizer.encode(prompt_text)

        context.extend(self.write('{"prompt": '))
        context.extend(self.write(f'"{prompt_obj.prompt}"'))
        context.extend(self.write(', "name": "'))

        generated: List[int] = []
        while True:
            logits = logits_fn(context + generated)
            token_id = self.predict(
                logits, self._fn_name_validator, generated
            )
            generated.append(token_id)
            print(
                self._tokenizer.decode(token_id),
                end="", flush=True,
            )
            if not self._fn_name_validator.get_valid_next_tokens(
                generated
            ):
                break

        context.extend(generated)

        fn_name = self._tokenizer.decode(generated)
        fn_obj = self._function_defs[fn_name]

        context.extend(self.write('", "parameters": {'))

        params = list(fn_obj.parameters.items())
        for i, (key, spec) in enumerate(params):
            context.extend(self.write(f'"{key}": '))

            param_type = spec.get("type", "string")
            validator: Any
            if param_type == "number":
                validator = self._number_validator
            elif param_type == "boolean":
                validator = self._boolean_validator
                validator.reset()
            else:
                validator = self._string_validator

            value_tokens: List[int] = []
            while True:
                logits = logits_fn(context + value_tokens)
                token_id = self.predict(
                    logits, validator, value_tokens
                )
                if validator.is_end(token_id):
                    break
                value_tokens.append(token_id)
                print(
                    self._tokenizer.decode(token_id),
                    end="", flush=True,
                )

            context.extend(value_tokens)

            if i < len(params) - 1:
                context.extend(self.write(", "))

        self.write("}}")
        return ""
