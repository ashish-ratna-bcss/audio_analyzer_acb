"""Unit tests for the eval runner aggregation + per-engine breakdown."""
import json
import os

from eval import reference_store, runner


def _write_case(root, case_id, reference, result):
    d = os.path.join(root, case_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "reference.txt"), "w", encoding="utf-8") as f:
        f.write(reference)
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


def test_load_and_evaluate(tmp_path):
    root = str(tmp_path)
    _write_case(root, "case1", "ధర పదిహేను రూపాయలు", {
        "transcript": {"segments": [
            {"text": "ధర పదిహేను", "source_pass": "telugu_whisper"},
            {"text": "రూపాయలు", "source_pass": "indic_conformer"},
        ]}})
    _write_case(root, "case2", "1100 rupees", {
        "transcript": {"segments": [
            {"text": "1100 rupees", "source_pass": "whisper"},
        ]}})

    pairs = reference_store.load_dataset(root)
    assert len(pairs) == 2
    rep = runner.evaluate(pairs, glossary={})

    assert rep["files_scored"] == 2
    assert rep["files_errored"] == 0
    assert set(rep["aggregate"]) == {"wer", "cer", "sfr", "number_acc", "entity_acc"}
    # both engines from case1 + whisper from case2 present
    assert {"telugu_whisper", "indic_conformer", "whisper"} <= set(rep["by_engine"])
    # case2 number preserved
    c2 = next(f for f in rep["per_file"] if f["case_id"] == "case2")
    assert c2["number_acc"] == 1.0


def test_missing_reference_recorded(tmp_path):
    root = str(tmp_path)
    d = os.path.join(root, "bad")
    os.makedirs(d)
    json.dump({"transcript": {"segments": [{"text": "x", "source_pass": "whisper"}]}},
              open(os.path.join(d, "result.json"), "w"))
    pairs = reference_store.load_dataset(root)
    rep = runner.evaluate(pairs)
    assert rep["files_errored"] == 1
    assert rep["errors"][0]["error"] == "missing reference"


def test_reference_json_rows_shape(tmp_path):
    root = str(tmp_path)
    d = os.path.join(root, "c")
    os.makedirs(d)
    json.dump({"rows": [{"conversation": "హలో"}, {"conversation": "సార్"}]},
              open(os.path.join(d, "reference.json"), "w"))
    json.dump({"transcript": {"segments": [{"text": "హలో సార్", "source_pass": "whisper"}]}},
              open(os.path.join(d, "result.json"), "w"))
    pairs = reference_store.load_dataset(root)
    assert pairs[0]["reference"] == "హలో సార్"
    rep = runner.evaluate(pairs)
    assert rep["files_scored"] == 1
