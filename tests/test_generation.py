import json

import pandas as pd

from ai_code_detector.data.sources.generated import GeneratedSource
from ai_code_detector.generation.clients import Answer, GenerationError, Generator
from ai_code_detector.generation.extract import extract_code
from ai_code_detector.generation.prompts import PromptStyle, build_prompt, html_to_text
from ai_code_detector.generation.runner import GenerationRunner, Pricing, plan_jobs


class _FakeGenerator(Generator):
    def __init__(self, name, answer=None, fail=False):
        super().__init__(name, f"{name}-model", max_tokens=100)
        self.answer, self.fail, self.calls = answer, fail, 0

    def _call(self, prompt):
        self.calls += 1
        if self.fail:
            raise GenerationError("quota")
        return Answer(self.answer, f"{self.name}-served", input_tokens=1000, output_tokens=500)


def test_extract_tagged_aliases_untagged_and_thinking():
    answer = ("<think>```python\nnot this\n```</think>\n"
              "```HTML\n<p>a</p>\n```\n```js\nlet x = 1;\n```\n```js\nlet y = 2;\n```")
    assert extract_code(answer, ["html", "javascript"]) == {
        "html": "<p>a</p>", "javascript": "let x = 1;\n\nlet y = 2;",
    }
    assert extract_code("```\nprint(1)\n```", ["python"]) == {"python": "print(1)"}
    assert extract_code("```\nprint(1)\n```", ["html", "css"]) == {}
    stray = "```html\n```java\nclass Main {}\n```"
    assert extract_code(stray, ["java"]) == {"java": "class Main {}"}
    assert extract_code("```java\nclass Main {", ["java"]) == {}


def test_embedded_css_and_js_are_moved_to_their_own_blocks():
    answer = ("```html\n<head><style>p { color: red; }</style></head>\n"
              "<body><p>x</p><script src=\"a.js\"></script>"
              "<script>let a = 1;</script></body>\n```")
    codes = extract_code(answer, ["html", "css", "javascript"])
    assert codes["css"] == "p { color: red; }"
    assert codes["javascript"] == "let a = 1;"
    assert "<style" not in codes["html"] and 'src="a.js"' in codes["html"]
    assert "css" not in extract_code(answer, ["html"])


def test_answer_cut_by_token_limit_is_rejected(monkeypatch):
    from types import SimpleNamespace

    import pytest

    from ai_code_detector.generation.clients import OpenAICompatibleGenerator

    monkeypatch.setenv("FAKE_KEY", "x")
    generator = OpenAICompatibleGenerator("g", "m", 100, "FAKE_KEY", reasoning_effort="low")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content="```java\nclass Main {")
        return SimpleNamespace(model="m", choices=[SimpleNamespace(finish_reason="length",
                                                                   message=message)])

    generator.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=create)))
    prompt = build_prompt("p:1", None, "s", "markdown", ["java"], PromptStyle.STANDARD, 100)
    with pytest.raises(GenerationError, match="cut"):
        generator.generate(prompt)
    assert calls[0]["reasoning_effort"] == "low"


def test_html_to_text_keeps_code_examples_and_drops_style():
    html = ("<style>p{}</style><h2>Loops</h2><p>Use <b>for</b>:</p>"
            "<pre><code>for i in x:\n  pass</code></pre>")
    assert html_to_text(html) == "Loops\n\nUse for:\n\nfor i in x:\n  pass"


def test_prompt_mentions_languages_style_and_truncation():
    prompt = build_prompt("webdproc:lesson:1", "Intro", "<p>" + "a" * 50 + "</p>", "html",
                          ["html", "css"], PromptStyle.NO_COMMENTS, max_statement_chars=10)
    assert prompt.truncated
    assert "HTML, CSS" in prompt.user and "lesson" in prompt.user
    assert "Do not write any comments" in prompt.user
    assert "separate HTML, CSS and JavaScript panes" in prompt.user
    single = build_prompt("p:1", None, "Sort", "markdown", ["java"], PromptStyle.STANDARD, 100)
    assert "panes" not in single.user and "Write the solution in Java." in single.user


def _tables():
    records = pd.DataFrame({
        "task_id": ["t1", "t1", "t2", "t3", "t4", "t5", "t5", "t5"],
        "language": ["html", "css", "python", "java", "python", "c", "java", "python"],
        "label": ["human", "human", "human", "human", "ai", "human", "human", "human"],
    })
    tasks = pd.DataFrame({
        "task_id": ["t1", "t2", "t3", "t4", "t5"], "dataset": ["w", "p", "p", "p", "p"],
        "title": ["T1", "T2", "T3", "T4", "T5"], "statement": ["<p>s</p>", "s", "", "s", "s"],
        "statement_format": ["html", "markdown", "markdown", "markdown", "markdown"],
    })
    return records, tasks


def test_generator_stops_when_budget_is_reached(tmp_path):
    records, tasks = _tables()
    generator = _FakeGenerator("paid", answer="```python\nprint(1)\n```")
    jobs = plan_jobs(records, tasks, ["paid"], [PromptStyle.STANDARD])
    # One solution costs (1000 * 2 + 500 * 10) / 1e6 = $0.007; budget allows one.
    pricing = {"paid": Pricing(2.0, 10.0, budget_usd=0.005)}
    summary = GenerationRunner({"paid": generator}, tasks, tmp_path, 1000, pricing).run(jobs)

    assert summary["paid"]["written"] == 1 and summary["paid"]["skipped"] == 2
    assert summary["paid"]["spent_usd"] == 0.007
    again = GenerationRunner({"paid": generator}, tasks, tmp_path, 1000, pricing).run(jobs)
    assert again["paid"]["written"] == 0 and generator.calls == 1


def test_exhaustive_plan_covers_every_style_and_language_and_keeps_default_jobs():
    records, tasks = _tables()
    styles = list(PromptStyle)
    default = plan_jobs(records, tasks, ["g1"], styles)
    full = plan_jobs(records, tasks, ["g1"], styles, exhaustive_datasets=["p"])

    t5 = [j for j in full if j.task_id == "t5"]
    assert {j.languages for j in t5} == {("c",), ("java",), ("python",)}
    assert len(t5) == 3 * len(styles)
    assert len([j for j in full if j.task_id == "t1"]) == 1
    assert {j.job_id for j in default} <= {j.job_id for j in full}


def test_plan_uses_human_languages_and_skips_tasks_without_statement():
    records, tasks = _tables()
    styles = list(PromptStyle)
    jobs = plan_jobs(records, tasks, ["g1", "g2", "g3", "g4"], styles)

    assert {j.task_id for j in jobs} == {"t1", "t2", "t5"}
    assert {j.languages for j in jobs if j.task_id == "t1"} == {("css", "html")}
    t5 = [j.languages for j in jobs if j.task_id == "t5"]
    assert all(len(langs) == 1 and langs[0] in {"c", "java", "python"} for langs in t5)
    assert plan_jobs(records, tasks, ["g1", "g2", "g3", "g4"], styles) == jobs
    assert len(plan_jobs(records, tasks, ["g1"], styles, limit_tasks=1)) == 1


def test_run_writes_results_resumes_and_feeds_generated_source(tmp_path):
    records, tasks = _tables()
    ok = _FakeGenerator("good", answer="```html\n<p>x</p>\n```\n```css\np { }\n```")
    bad = _FakeGenerator("bad", fail=True)
    jobs = plan_jobs(records, tasks, ["good", "bad"], [PromptStyle.STANDARD])
    runner = GenerationRunner({"good": ok, "bad": bad}, tasks, tmp_path, 1000)

    summary = runner.run(jobs)
    assert summary["good"]["written"] == 3 and summary["bad"]["failed"] == 3
    assert not (tmp_path / "bad.jsonl").exists()

    again = runner.run(jobs)
    assert again["good"]["done_before"] == 3 and ok.calls == 3

    rows = [json.loads(line) for line in (tmp_path / "good.jsonl").read_text().splitlines()]
    assert rows[0]["served_model"] == "good-served"
    assert rows[0]["usage"] == {"input_tokens": 1000, "output_tokens": 500}

    generated = list(GeneratedSource(tmp_path).load())
    t1 = {r.language: r for r in generated if r.task_id == "t1"}
    assert set(t1) == {"html", "css"}
    assert t1["html"].label == "ai" and t1["html"].label_status == "verified"
    assert t1["html"].generator == "good" and t1["html"].prompt_style == "standard"
