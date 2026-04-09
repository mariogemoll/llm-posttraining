from typing import Any, Sequence

class SamplingParams:
    def __init__(self, temperature: float = ..., max_tokens: int = ..., **kwargs: Any) -> None: ...

class RequestOutput:
    class CompletionOutput:
        text: str
        token_ids: Sequence[int]
    outputs: Sequence[CompletionOutput]

class LLM:
    def __init__(
        self,
        model: str,
        dtype: str = ...,
        enable_lora: bool = ...,
        max_model_len: int = ...,
        **kwargs: Any,
    ) -> None: ...
    def generate(
        self,
        prompts: Sequence[str],
        sampling_params: SamplingParams,
        lora_request: Any = ...,
    ) -> list[RequestOutput]: ...
