"""Clients that send a prompt to a language model and return its text answer."""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import anthropic
import openai

from ai_code_detector.generation.prompts import Prompt


class GenerationError(RuntimeError):
    """The generator did not return a usable answer (refusal, empty output, API error)."""


@dataclass(frozen=True)
class Answer:
    """Text returned by a generator, the model that produced it and the tokens it billed."""

    text: str
    served_model: str
    input_tokens: int = 0
    output_tokens: int = 0


class Generator(ABC):
    """One named model behind one provider API."""

    def __init__(self, name: str, model: str, max_tokens: int, min_interval: float = 0.0) -> None:
        """Store the identity and limits shared by all providers."""
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.min_interval = min_interval
        self._last_call = 0.0

    def generate(self, prompt: Prompt) -> Answer:
        """Send the prompt, respecting the minimal pause between calls of this generator."""
        wait = self._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return self._call(prompt)
        finally:
            self._last_call = time.monotonic()

    @abstractmethod
    def _call(self, prompt: Prompt) -> Answer:
        """Provider-specific request."""


class AnthropicGenerator(Generator):
    """Claude through the Anthropic Messages API.

    Server-side refusal fallback is enabled; the answer records which model served it.
    """

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, name: str, model: str, max_tokens: int, api_key_env: str,
                 effort: str = "low", min_interval: float = 0.0) -> None:
        """Create the client from the API key stored in the given environment variable."""
        super().__init__(name, model, max_tokens, min_interval)
        self.effort = effort
        self.client = anthropic.Anthropic(api_key=os.environ[api_key_env], max_retries=3)

    def _call(self, prompt: Prompt) -> Answer:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
                output_config={"effort": self.effort},
                betas=[self.FALLBACK_BETA],
                extra_body={"fallbacks": "default"},
            )
        except anthropic.APIError as error:
            raise GenerationError(f"{self.name}: {error}") from error
        if response.stop_reason == "refusal":
            raise GenerationError(f"{self.name}: refused")
        if response.stop_reason == "max_tokens":
            raise GenerationError(f"{self.name}: answer cut at max_tokens")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text.strip():
            raise GenerationError(f"{self.name}: empty answer ({response.stop_reason})")
        usage = response.usage
        return Answer(text, response.model, usage.input_tokens, usage.output_tokens)


class OpenAICompatibleGenerator(Generator):
    """Any provider exposing the OpenAI Chat Completions API (Gemini, Groq, OpenAI)."""

    def __init__(self, name: str, model: str, max_tokens: int, api_key_env: str,
                 base_url: str | None = None, min_interval: float = 0.0,
                 reasoning_effort: str | None = None) -> None:
        """Create the client from the API key stored in the given environment variable."""
        super().__init__(name, model, max_tokens, min_interval)
        self.reasoning_effort = reasoning_effort
        self.client = openai.OpenAI(api_key=os.environ[api_key_env], base_url=base_url,
                                    max_retries=3, timeout=180.0)

    def _call(self, prompt: Prompt) -> Answer:
        options = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                **options,
            )
        except openai.APIError as error:
            raise GenerationError(f"{self.name}: {error}") from error
        if not response.choices:
            raise GenerationError(f"{self.name}: no choices in answer")
        choice = response.choices[0]
        # Reasoning models may spend the token budget on hidden reasoning and cut the code.
        if choice.finish_reason == "length":
            raise GenerationError(f"{self.name}: answer cut at max_tokens")
        text = choice.message.content or ""
        if not text.strip():
            raise GenerationError(f"{self.name}: empty answer")
        return Answer(text, response.model or self.model, *self._billed_tokens(response.usage))

    @staticmethod
    def _billed_tokens(usage) -> tuple[int, int]:
        """Input and output tokens to bill.

        Some providers leave hidden reasoning out of ``completion_tokens`` but still bill
        it; it shows up in ``total_tokens``, so output is taken as total minus input.
        """
        if usage is None:
            return 0, 0
        prompt = usage.prompt_tokens or 0
        output = max(usage.completion_tokens or 0, (usage.total_tokens or 0) - prompt)
        return prompt, output


def build_generator(name: str, settings: dict[str, Any], max_tokens: int) -> Generator:
    """Create a generator from its config entry; its ``max_tokens`` overrides the default."""
    provider = settings["provider"]
    common = {
        "name": name,
        "model": settings["model"],
        "max_tokens": int(settings.get("max_tokens", max_tokens)),
        "api_key_env": settings["api_key_env"],
        "min_interval": float(settings.get("min_interval", 0.0)),
    }
    if provider == "anthropic":
        return AnthropicGenerator(**common, effort=settings.get("effort", "low"))
    if provider == "openai_compatible":
        return OpenAICompatibleGenerator(**common, base_url=settings.get("base_url"),
                                         reasoning_effort=settings.get("reasoning_effort"))
    raise ValueError(f"Unknown provider {provider!r} for generator {name!r}")
