"""Prompts sent to the generators: task statement plus a writing style."""

from dataclasses import dataclass
from enum import StrEnum
from html.parser import HTMLParser

WEB_LANGUAGES = frozenset({"html", "css", "javascript"})

# Human web code comes from an editor with separate HTML, CSS and JS panes, so AI code
# must be split the same way or the split itself would become a distinguishing feature.
WEB_SPLIT_INSTRUCTION = (
    "The editor has separate HTML, CSS and JavaScript panes: put each language in its own "
    "code block and do not embed CSS or JavaScript inside the HTML."
)

LANGUAGE_NAMES = {
    "html": "HTML", "css": "CSS", "javascript": "JavaScript", "python": "Python",
    "c": "C", "cpp": "C++", "csharp": "C#", "php": "PHP", "java": "Java",
}

SYSTEM_PROMPT = (
    "You write source code. Answer with code only: one fenced code block per requested "
    "language, tagged with the language name (for example ```html, ```css, ```javascript, "
    "```python, ```cpp, ```csharp, ```php, ```java, ```c). No text outside the code blocks."
)


class PromptStyle(StrEnum):
    """How the generator is asked to write; varied so the AI class is not one style."""

    STANDARD = "standard"
    BEGINNER = "beginner"
    NO_COMMENTS = "no_comments"


STYLE_INSTRUCTIONS = {
    PromptStyle.STANDARD: "",
    PromptStyle.BEGINNER: "Write it the way a beginner programming student would.",
    PromptStyle.NO_COMMENTS: "Do not write any comments in the code.",
}

_BLOCK_TAGS = {"p", "div", "section", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4",
               "h5", "h6", "tr", "table", "pre", "blockquote"}


class _TextExtractor(HTMLParser):
    """Turn HTML into plain text, keeping line breaks and the content of code examples."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Convert an HTML statement to readable plain text."""
    parser = _TextExtractor()
    parser.feed(html)
    lines = [line.rstrip() for line in "".join(parser.parts).splitlines()]
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


@dataclass(frozen=True)
class Prompt:
    """A ready-to-send prompt and whether the statement had to be shortened."""

    system: str
    user: str
    truncated: bool


def build_prompt(task_id: str, title: str | None, statement: str, statement_format: str,
                 languages: list[str], style: PromptStyle, max_statement_chars: int) -> Prompt:
    """Build the prompt asking for a solution of one task in the given languages."""
    text = html_to_text(statement) if statement_format == "html" else statement.strip()
    truncated = len(text) > max_statement_chars
    text = text[:max_statement_chars]

    names = ", ".join(LANGUAGE_NAMES.get(lang, lang) for lang in languages)
    if ":lesson:" in task_id:
        intro = "This is the material of a lesson from a programming course."
        ask = f"Write the {names} code you would write as practice work for this lesson."
    else:
        intro = "Solve this programming task."
        ask = f"Write the solution in {names}."
    header = f"{intro}\n\nTitle: {title}" if title else intro
    split = WEB_SPLIT_INSTRUCTION if len(languages) > 1 and set(languages) <= WEB_LANGUAGES else ""
    parts = [header, text, ask, split, STYLE_INSTRUCTIONS[style]]
    return Prompt(SYSTEM_PROMPT, "\n\n".join(p for p in parts if p), truncated)
