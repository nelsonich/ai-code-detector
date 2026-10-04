"""Signals from a code language model: how predictable a sample is to it.

AI generators pick likely tokens, so their code tends to be more predictable to another
language model than human code (DetectGPT4Code, DetectCodeGPT). Binoculars (Hans et al.,
2024) divides the perplexity by the cross-perplexity between two related models, which
cancels how predictable the task itself is. None of these signals needs labelled data.
"""

import re

_BLANK_RUN = re.compile(r"\n{3,}")


def normalize_layout(code: str) -> str:
    """Remove layout an editor or formatter can change: trailing spaces, tabs, blank runs."""
    lines = [line.rstrip().expandtabs(4) for line in code.strip("\n").split("\n")]
    return _BLANK_RUN.sub("\n\n", "\n".join(lines))


class LanguageModelScorer:
    """Score code with an observer and a performer model that share one tokenizer.

    The observer is a base model and the performer its instruction-tuned version, as in
    Binoculars. Code longer than ``max_tokens`` is cut, which keeps CPU time bounded.
    """

    def __init__(self, observer: str, performer: str, max_tokens: int = 512,
                 threads: int | None = None) -> None:
        """Load both models on the CPU."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(observer)
        self.observer = AutoModelForCausalLM.from_pretrained(observer, dtype=torch.float32)
        self.performer = AutoModelForCausalLM.from_pretrained(performer, dtype=torch.float32)
        self.observer.eval()
        self.performer.eval()

    def score(self, code: str) -> dict[str, float]:
        """Return the language-model signals of one code sample."""
        torch = self._torch
        ids = self.tokenizer(code, return_tensors="pt", truncation=True,
                             max_length=self.max_tokens)["input_ids"]
        if ids.shape[1] < 2:
            return dict.fromkeys(SIGNALS, float("nan"))
        with torch.no_grad():
            observer_logits = self.observer(ids).logits[0, :-1].float()
            performer_logits = self.performer(ids).logits[0, :-1].float()
        targets = ids[0, 1:]
        observer_logp = torch.log_softmax(observer_logits, dim=-1)
        performer_logp = torch.log_softmax(performer_logits, dim=-1)
        token_logp = observer_logp.gather(1, targets[:, None])[:, 0]
        observer_p = observer_logp.exp()
        entropy = -(observer_p * observer_logp).sum(-1)
        performer_nll = -performer_logp.gather(1, targets[:, None])[:, 0].mean()
        cross_entropy = -(observer_p * performer_logp).sum(-1).mean()
        top1 = (observer_logits.argmax(-1) == targets).float()
        return {
            "lm_logprob_mean": float(token_logp.mean()),
            "lm_logprob_std": float(token_logp.std()) if len(targets) > 1 else 0.0,
            "lm_entropy_mean": float(entropy.mean()),
            "lm_top1_ratio": float(top1.mean()),
            "lm_binoculars": float(performer_nll / cross_entropy),
            "lm_tokens": float(len(targets) + 1),
        }


SIGNALS = ("lm_logprob_mean", "lm_logprob_std", "lm_entropy_mean", "lm_top1_ratio",
           "lm_binoculars", "lm_tokens")

