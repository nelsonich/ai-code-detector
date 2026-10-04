import pandas as pd

from ai_code_detector.data.sources.droid import DroidSource, origin_of


def _write(root, rows):
    root.mkdir()
    pd.DataFrame(rows).to_parquet(root / "test.parquet")


def _row(label, source, language="Python", generator="Human"):
    return {"Code": f"print('{label} {source}')", "Generator": generator,
            "Generation_Mode": "INSTRUCT", "Source": source, "Language": language,
            "Sampling_Params": "", "Rewriting_Params": "", "Label": label,
            "Model_Family": generator}


def test_origin_groups_sources_into_populations():
    assert origin_of("CODEFORCES") == "droid-cp"
    assert origin_of("THEVAULT_FUNCTION") == "droid-github"
    assert origin_of("DROID_PERSONAHUB") is None


def test_keeps_human_and_ai_of_known_languages_and_origins(tmp_path):
    root = tmp_path / "droid"
    _write(root, [
        _row("HUMAN_GENERATED", "LEETCODE"),
        _row("MACHINE_GENERATED", "LEETCODE", generator="GPT-4o"),
        _row("MACHINE_REFINED", "LEETCODE", generator="GPT-4o"),
        _row("HUMAN_GENERATED", "LEETCODE", language="Go"),
        _row("MACHINE_GENERATED", "DROID_PERSONAHUB", generator="GPT-4o"),
    ])
    records = sorted(DroidSource(root).load(), key=lambda r: r.label)
    assert [(r.label, r.origin, r.language) for r in records] == [
        ("ai", "droid-cp", "python"), ("human", "droid-cp", "python")]
    assert records[0].generator == "GPT-4o" and records[0].prompt_style == "instruct"
    assert records[1].generator is None


def test_samples_at_most_per_group(tmp_path):
    root = tmp_path / "droid"
    _write(root, [_row("HUMAN_GENERATED", "TACO") for _ in range(5)])
    assert len(list(DroidSource(root, per_group=2).load())) == 2
