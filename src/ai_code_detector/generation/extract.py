"""Extraction of code blocks from a generator's answer."""

import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_STYLE = re.compile(r"<style[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)
_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.DOTALL | re.IGNORECASE)

ALIASES = {
    "html": "html", "htm": "html",
    "css": "css",
    "javascript": "javascript", "js": "javascript",
    "python": "python", "py": "python", "python3": "python",
    "c": "c",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp",
    "csharp": "csharp", "cs": "csharp", "c#": "csharp",
    "php": "php",
    "java": "java",
}


def _fenced_blocks(answer: str) -> list[tuple[str, str]]:
    """Split an answer into (tag, body) pairs of closed fenced blocks.

    A fence with a tag that arrives while the open block is still empty replaces the
    tag instead of closing the block, so an empty ```html fence followed by ```java
    yields one java block.
    An unclosed last block is dropped: the answer was cut and the code is incomplete.
    """
    blocks: list[tuple[str, str]] = []
    tag: str | None = None
    body: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped.startswith("```"):
            if tag is not None:
                body.append(line)
            continue
        fence_tag = stripped[3:].split()[0] if stripped[3:].strip() else ""
        if tag is None:
            tag, body = fence_tag, []
        elif fence_tag and not "".join(body).strip():
            tag = fence_tag
        else:
            blocks.append((tag, "\n".join(body)))
            tag, body = None, []
    return blocks


def extract_code(answer: str, languages: list[str]) -> dict[str, str]:
    """Return the code for each requested language found in the answer.

    Blocks tagged with an unknown or missing language are assigned only when exactly
    one language was requested. Several blocks of one language are joined. When CSS or
    JavaScript was requested but came embedded in the HTML, it is moved to its own block.
    """
    found: dict[str, list[str]] = {}
    for tag, body in _fenced_blocks(_THINK.sub("", answer)):
        language = ALIASES.get(tag.lower())
        if language is None and len(languages) == 1:
            language = languages[0]
        if language in languages and body.strip():
            found.setdefault(language, []).append(body.rstrip())
    codes = {language: "\n\n".join(blocks) for language, blocks in found.items()}
    if "html" in codes and "html" in languages:
        _split_embedded(codes, languages)
    return codes


def _split_embedded(codes: dict[str, str], languages: list[str]) -> None:
    for language, pattern in (("css", _STYLE), ("javascript", _SCRIPT)):
        if language not in languages or language in codes:
            continue
        embedded = [body.strip() for body in pattern.findall(codes["html"]) if body.strip()]
        if embedded:
            codes[language] = "\n\n".join(embedded)
            codes["html"] = pattern.sub("", codes["html"]).strip()
