from ai_code_detector.data.schema import (
    RECORD_COLUMNS,
    CodeRecord,
    Label,
    LabelStatus,
    SourceKind,
    records_to_frame,
)


def test_frame_stores_enums_as_plain_strings():
    record = CodeRecord(
        record_id="r1", code="x = 1", language="python", label=Label.HUMAN,
        label_status=LabelStatus.VERIFIED, source=SourceKind.EXTERNAL,
        task_id="t1", dataset="progpedia",
    )
    frame = records_to_frame([record])
    assert list(frame.columns) == RECORD_COLUMNS
    assert frame.loc[0, "label"] == "human"
    assert frame.loc[0, "label_status"] == "verified"
    assert frame.loc[0, "source"] == "external"
