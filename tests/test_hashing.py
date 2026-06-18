import hashlib
from services.hashing import sha256_bytes, sha256_file


def test_sha256_bytes_matches_hashlib():
    data = b"forensic evidence bytes"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_file_streams(tmp_path):
    p = tmp_path / "a.bin"
    payload = b"x" * (1024 * 1024 + 7)  # > one chunk
    p.write_bytes(payload)
    assert sha256_file(str(p)) == hashlib.sha256(payload).hexdigest()
