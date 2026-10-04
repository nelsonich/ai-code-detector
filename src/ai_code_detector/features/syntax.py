"""Lightweight, regex-based knowledge of the supported languages' syntax.

Not a parser: comments, strings and identifiers are found approximately, which is
enough for style statistics over thousands of samples.
"""

import re

C_LIKE = {"javascript", "c", "cpp", "csharp", "java", "php"}

_BLOCK_COMMENT = r"/\*.*?\*/"
# "//" not preceded by ":" or a quote, so "http://..." inside code is not a comment.
_LINE_COMMENT = r"(?<![:\"'])//[^\n]*"
_HASH_COMMENT = r"#[^\n]*"
_HTML_COMMENT = r"<!--.*?-->"

_COMMENT_PATTERNS = {
    **{language: [_BLOCK_COMMENT, _LINE_COMMENT] for language in C_LIKE},
    "php": [_BLOCK_COMMENT, _LINE_COMMENT, r"(?<![\w\[$\"'])#(?!\[)[^\n]*"],
    "css": [_BLOCK_COMMENT],
    "python": [_HASH_COMMENT],
    "html": [_HTML_COMMENT],
}

_STRING = re.compile(r"\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`", re.DOTALL)
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_TOKEN = re.compile(r"\w+|[^\w\s]")

_KEYWORD_TEXT = {
    "javascript": "break case catch class const continue default delete do else export extends "
                  "false finally for function if import in instanceof let new null return super "
                  "switch this throw true try typeof undefined var void while async await of",
    "python": "and as assert break class continue def del elif else except False finally for "
              "from global if import in is lambda None nonlocal not or pass raise return True "
              "try while with yield print range len self",
    "c": "auto break case char const continue default do double else enum extern float for "
         "goto if int long register return short signed sizeof static struct switch typedef "
         "union unsigned void volatile while include define printf scanf main",
    "cpp": "auto bool break case catch char class const continue default delete do double else "
           "enum false float for if include int long namespace new nullptr private protected "
           "public return short signed sizeof static std struct switch template this true try "
           "typename unsigned using void while cin cout endl string vector main",
    "csharp": "abstract as base bool break case catch char class const continue decimal default "
              "do double else enum false finally float for foreach if in int interface internal "
              "is long namespace new null object out override private protected public readonly "
              "return static string struct switch this throw true try using var void while "
              "Console WriteLine ReadLine Main System",
    "java": "abstract boolean break byte case catch char class continue default do double else "
            "extends false final finally float for if implements import int interface long new "
            "null package private protected public return short static super switch this throw "
            "throws true try void while String System out println main var",
    "php": "abstract and array as break case catch class const continue default do echo else "
           "elseif empty extends false finally for foreach function global if implements isset "
           "new null or private protected public return static switch this throw true try "
           "while php",
    "css": "",
    "html": "",
}
KEYWORDS = {language: frozenset(words.split()) for language, words in _KEYWORD_TEXT.items()}

# Constructs typical of current textbook style ("modern") and of older or ad-hoc code
# ("legacy"). Counted per line; only comparable within one language.
CONSTRUCTS = {
    "javascript": (
        [r"\bconst\b", r"\blet\b", r"=>", r"`", r"===|!==", r"\b(?:async|await)\b",
         r"\.(?:map|filter|reduce|forEach)\("],
        [r"\bvar\b", r"[^=!<>]==[^=]", r"[^!]!=[^=]"],
    ),
    "python": (
        [r"\bf[\"']", r"->", r":=", r"\bwith\b", r"\b(?:enumerate|zip)\("],
        [r"%\s*\(", r"\brange\(len\(", r"\.has_key\("],
    ),
    "c": (
        [r"\bbool\b", r"\bfor\s*\(\s*int\b", r"\bconst\b"],
        [r"\bgoto\b", r"#define\b"],
    ),
    "cpp": (
        [r"\bauto\b", r"\bnullptr\b", r"\bfor\s*\([^;)]*:[^;)]*\)", r"\bconst\b",
         r"\bstd::(?:vector|string|map|unordered_map)\b"],
        [r"\bNULL\b", r"#define\b", r"\bmalloc\(", r"\bprintf\("],
    ),
    "csharp": (
        [r"\bvar\b", r"=>", r"\$\"", r"\?\.", r"\.(?:Select|Where|ToList)\("],
        [r"string\.Format\(", r"\bArrayList\b"],
    ),
    "java": (
        [r"\bvar\b", r"->", r"\.stream\(\)", r"\bList\.of\(", r"@Override"],
        [r"\bVector\b", r"\bStringBuffer\b"],
    ),
    "php": (
        [r"\?\?", r"\bfn\s*\(", r"\bmatch\s*\(", r"=\s*\[", r"\bdeclare\(strict_types"],
        [r"\barray\s*\(", r"\bmysql_"],
    ),
    "css": (
        [r"\bvar\(--", r"display\s*:\s*(?:flex|grid)", r"\d\s*rem\b", r"@media"],
        [r"\bfloat\s*:", r"!important"],
    ),
    "html": (
        [r"<(?:header|nav|main|section|article|footer|aside)\b",
         r"<meta[^>]+name=[\"']viewport", r"<!DOCTYPE\s+html>"],
        [r"<(?:center|font|marquee)\b", r"\bbgcolor=", r"<table[^>]*\bborder="],
    ),
}
_COMPILED = {
    language: tuple([re.compile(p, re.IGNORECASE if language == "html" else 0) for p in group]
                    for group in groups)
    for language, groups in CONSTRUCTS.items()
}


def split_comments(code: str, language: str) -> tuple[list[str], str]:
    """Return the comments found in ``code`` and the code with comments blanked out."""
    comments: list[str] = []
    for pattern in _COMMENT_PATTERNS.get(language, []):
        regex = re.compile(pattern, re.DOTALL)
        comments.extend(regex.findall(code))
        code = regex.sub(lambda m: "\n" * m.group(0).count("\n"), code)
    return comments, code


def strip_strings(code: str) -> str:
    """Replace string literals with empty quotes."""
    return _STRING.sub('""', code)


def identifiers(code: str, language: str) -> list[str]:
    """Return identifiers of comment- and string-free code, without keywords."""
    keywords = KEYWORDS.get(language, frozenset())
    return [name for name in _IDENTIFIER.findall(code) if name not in keywords]


def tokens(code: str) -> list[str]:
    """Split code into words and single punctuation characters."""
    return _TOKEN.findall(code)


def construct_counts(code: str, language: str) -> tuple[int, int]:
    """Count modern and legacy constructs of the language."""
    modern, legacy = _COMPILED.get(language, ([], []))
    return (sum(len(p.findall(code)) for p in modern),
            sum(len(p.findall(code)) for p in legacy))
