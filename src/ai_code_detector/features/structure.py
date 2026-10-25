"""Program-structure features from a real syntax tree (tree-sitter).

Unlike the regex features, the tree knows what is a statement, a block, a function or
a string, so keywords in docstrings, braces of data literals or wrapped lines do not
count as structure. The features describe the logic of a program and do not change
when an editor re-indents or re-spaces it.
"""

from functools import cache

import tree_sitter as ts

_C_LOOPS = {"if_statement", "for_statement", "while_statement", "do_statement",
            "conditional_expression"}
_LOGICAL = {"&&", "||", "??", "and", "or"}


class _Grammar:
    """Node types that matter for one language."""

    def __init__(self, module_language, blocks, decisions, functions, statements=None) -> None:
        self.language = ts.Language(module_language)
        self.blocks = frozenset(blocks)
        self.decisions = frozenset(decisions)
        self.functions = frozenset(functions)
        self.statements = frozenset(statements or ())


@cache
def _grammars() -> dict[str, _Grammar]:
    import tree_sitter_c
    import tree_sitter_c_sharp
    import tree_sitter_cpp
    import tree_sitter_css
    import tree_sitter_html
    import tree_sitter_java
    import tree_sitter_javascript
    import tree_sitter_php
    import tree_sitter_python

    return {
        "python": _Grammar(
            tree_sitter_python.language(), {"block"},
            {"if_statement", "elif_clause", "for_statement", "while_statement",
             "except_clause", "conditional_expression", "for_in_clause", "if_clause",
             "boolean_operator", "case_clause"},
            {"function_definition"}),
        "javascript": _Grammar(
            tree_sitter_javascript.language(), {"statement_block"},
            {"if_statement", "for_statement", "for_in_statement", "while_statement",
             "do_statement", "switch_case", "catch_clause", "ternary_expression"},
            {"function_declaration", "function_expression", "arrow_function",
             "method_definition", "generator_function_declaration"}),
        "c": _Grammar(
            tree_sitter_c.language(), {"compound_statement"},
            _C_LOOPS | {"case_statement"}, {"function_definition"}),
        "cpp": _Grammar(
            tree_sitter_cpp.language(), {"compound_statement"},
            _C_LOOPS | {"for_range_loop", "case_statement", "catch_clause"},
            {"function_definition", "lambda_expression"}),
        "csharp": _Grammar(
            tree_sitter_c_sharp.language(), {"block"},
            _C_LOOPS | {"foreach_statement", "switch_section", "catch_clause",
                        "switch_expression_arm"},
            {"method_declaration", "constructor_declaration", "local_function_statement",
             "lambda_expression"}),
        "java": _Grammar(
            tree_sitter_java.language(), {"block"},
            {"if_statement", "for_statement", "enhanced_for_statement", "while_statement",
             "do_statement", "switch_block_statement_group", "catch_clause",
             "ternary_expression"},
            {"method_declaration", "constructor_declaration", "lambda_expression"}),
        "php": _Grammar(
            tree_sitter_php.language_php(), {"compound_statement"},
            _C_LOOPS | {"else_if_clause", "foreach_statement", "case_statement",
                        "catch_clause"},
            {"function_definition", "method_declaration", "anonymous_function",
             "arrow_function"}),
        "html": _Grammar(
            tree_sitter_html.language(), {"element", "script_element", "style_element"},
            set(), set(), {"element", "script_element", "style_element"}),
        "css": _Grammar(
            tree_sitter_css.language(), {"block"}, set(), set(), {"declaration"}),
    }


FEATURES = ("max_nesting_depth", "avg_nesting_depth", "avg_cyclomatic_complexity",
            "max_cyclomatic_complexity", "decisions_per_statement", "statements_per_line",
            "avg_function_lines", "has_syntax_error")
_NOT_STATEMENTS = {"compound_statement", "statement_block"}


def _is_statement(node: ts.Node, grammar: _Grammar) -> bool:
    if grammar.statements:
        return node.type in grammar.statements
    return ((node.type.endswith("_statement") or node.type.endswith("declaration"))
            and node.type not in _NOT_STATEMENTS and "parameter" not in node.type)


def _is_decision(node: ts.Node, grammar: _Grammar) -> bool:
    if node.type in grammar.decisions:
        return True
    if node.type == "binary_expression":
        operator = node.child_by_field_name("operator")
        return operator is not None and operator.type in _LOGICAL
    return False


def extract(code: str, language: str) -> dict[str, float]:
    """Compute the structure features of one sample; zeros for unsupported languages."""
    grammar = _grammars().get(language)
    if grammar is None or not code.strip():
        return dict.fromkeys(FEATURES, 0.0)
    tree = ts.Parser(grammar.language).parse(code.encode("utf-8"))

    statement_depths: list[int] = []
    decisions_per_unit: dict[int, int] = {-1: 0}
    function_lines: list[int] = []
    stack = [(tree.root_node, 0, -1)]
    while stack:
        node, depth, unit = stack.pop()
        if node.type in grammar.functions:
            unit = len(function_lines)
            function_lines.append(node.end_point.row - node.start_point.row + 1)
            decisions_per_unit[unit] = 0
        if _is_decision(node, grammar):
            decisions_per_unit[unit] += 1
        if _is_statement(node, grammar):
            statement_depths.append(depth)
        child_depth = depth + (node.type in grammar.blocks)
        stack.extend((child, child_depth, unit) for child in node.named_children)

    # Cyclomatic complexity per function is its decisions + 1; top-level code counts
    # as one more unit when it has decisions or when there are no functions at all.
    units = [n for u, n in decisions_per_unit.items() if u >= 0]
    if decisions_per_unit[-1] or not units:
        units.append(decisions_per_unit[-1])
    complexity = [n + 1 for n in units]
    n_decisions = sum(units)
    lines = sum(1 for line in code.split("\n") if line.strip()) or 1
    return {
        "max_nesting_depth": float(max(statement_depths, default=0)),
        "avg_nesting_depth": (sum(statement_depths) / len(statement_depths)
                              if statement_depths else 0.0),
        "avg_cyclomatic_complexity": sum(complexity) / len(complexity),
        "max_cyclomatic_complexity": float(max(complexity)),
        "decisions_per_statement": n_decisions / max(len(statement_depths), 1),
        "statements_per_line": len(statement_depths) / lines,
        "avg_function_lines": (sum(function_lines) / len(function_lines)
                               if function_lines else 0.0),
        "has_syntax_error": float(tree.root_node.has_error),
    }
