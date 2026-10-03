from ai_code_detector.data.sources.codenet import CodenetSource

HEADER = "submission_id,problem_id,user_id,date,language,original_language,filename_ext,status\n"


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _submission(root, sid, problem, user, lang, ext, status="Accepted"):
    _write(root, f"data/{problem}/{lang}/{sid}.{ext}", f"code {sid}")
    return f"{sid},{problem},{user},1500000000,{lang},{lang},{ext},{status}\n"


def test_samples_accepted_unique_authors_with_limits(tmp_path):
    rows = [
        _submission(tmp_path, "s1", "p00001", "u1", "PHP", "php"),
        _submission(tmp_path, "s2", "p00001", "u1", "PHP", "php"),
        _submission(tmp_path, "s3", "p00001", "u2", "PHP", "php", status="Wrong Answer"),
        _submission(tmp_path, "s4", "p00001", "u3", "PHP", "php"),
        _submission(tmp_path, "s5", "p00001", "u4", "PHP", "php"),
        _submission(tmp_path, "s6", "p00001", "u5", "C#", "cs"),
    ]
    _write(tmp_path, "metadata/p00001.csv", HEADER + "".join(rows))
    _write(tmp_path, "metadata/problem_list.csv",
           "id,name,dataset,time_limit,memory_limit,rating,tags,complexity\n"
           "p00001,Sum,AIZU,1000,131072,,,\n")
    _write(tmp_path, "problem_descriptions/p00001.html", "<p>Add numbers</p>")

    records = list(CodenetSource(tmp_path, per_language=10, max_per_problem=2).load())
    php = [r for r in records if r.language == "php"]

    assert len(php) == 2
    assert len({r.author_id for r in php}) == 2
    assert {r.language for r in records} == {"php", "csharp"}
    assert all(r.label_status == "verified" and r.created_at.year == 2017 for r in records)


def test_missing_sampled_file_raises(tmp_path):
    row = _submission(tmp_path, "s1", "p00001", "u1", "PHP", "php")
    (tmp_path / "data/p00001/PHP/s1.php").unlink()
    _write(tmp_path, "metadata/p00001.csv", HEADER + row)
    _write(tmp_path, "metadata/problem_list.csv",
           "id,name,dataset,time_limit,memory_limit,rating,tags,complexity\n"
           "p00001,Sum,AIZU,1000,131072,,,\n")
    _write(tmp_path, "problem_descriptions/p00001.html", "<p>Add numbers</p>")

    try:
        list(CodenetSource(tmp_path).load())
    except FileNotFoundError as error:
        assert "download_data.sh" in str(error)
    else:
        raise AssertionError("expected FileNotFoundError")
