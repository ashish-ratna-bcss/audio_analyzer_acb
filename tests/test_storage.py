from services import storage, manifest_service as man
from services.hashing import sha256_file


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.config, "CASE_STORE_PATH", str(tmp_path))
    monkeypatch.setattr(man.config, "CASE_STORE_PATH", str(tmp_path))


def test_write_original_moves_hashes_and_locks(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    src = tmp_path / "staged.wav"
    src.write_bytes(b"RIFFfake")
    dest, digest = storage.write_original("c1", "f1", ".wav", str(src))
    assert dest.endswith("f1__original.wav")
    assert digest == sha256_file(dest)
    assert not src.exists()  # moved, not copied


def test_manifest_lineage(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    man.register_file("c1", "f1", "REC001.wav", "hashA")
    man.register_derivative("c1", "f1", "normalized_16k", "/p/16k.wav",
                            "hashB", parent_sha256="hashA")
    m = man.load("c1")
    assert m["files"]["f1"]["source_sha256"] == "hashA"
    deriv = m["files"]["f1"]["derivatives"][0]
    assert deriv["kind"] == "normalized_16k"
    assert deriv["parent_sha256"] == "hashA" and deriv["sha256"] == "hashB"
