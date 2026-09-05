import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "h1_h2_measurements", Path(__file__).parent.parent / "scripts" / "13_h1_h2_measurements.py"
)
assert _SPEC is not None and _SPEC.loader is not None
h1_h2_measurements = importlib.util.module_from_spec(_SPEC)
sys.modules["h1_h2_measurements"] = h1_h2_measurements
_SPEC.loader.exec_module(h1_h2_measurements)


def test_producer_receipt_binds_script_bytes(tmp_path):
    module_path = h1_h2_measurements.__file__
    assert module_path is not None
    path = Path(module_path).resolve()

    receipt = h1_h2_measurements.producer_code_receipt()

    assert receipt == {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_doc_acts(path: Path, ids: np.ndarray, *, layer: int = 22) -> str:
    config = '{"model_path":"olmo3-base","max_tokens":4096}'
    np.savez(
        path,
        ids=ids,
        acts=np.zeros((len(ids), 4), dtype=np.float16),
        layer=layer,
        config=config,
    )
    return config


def test_p1a_receipt_records_activation_provenance_and_id_order(tmp_path):
    ids = np.array(["doc-0", "doc-1", "doc-2"])
    path = tmp_path / "doc_acts.npz"
    config = write_doc_acts(path, ids)
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    acts, receipt = h1_h2_measurements.load_p1a_doc_acts(path, ids, 4, 22, expected_sha256)

    assert acts.shape == (3, 4)
    assert receipt == {
        "status": "consumed",
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "expected_sha256": expected_sha256,
        "sha256_matches_expected": True,
        "shape": [3, 4],
        "dtype": "float16",
        "layer": 22,
        "config": config,
        "doc_acts_ids_equal_doc_scores": True,
        "doc_acts_ids_order_equal_doc_scores": True,
        "n_documents": 3,
    }


def test_p1a_rejects_doc_acts_with_different_id_order(tmp_path):
    path = tmp_path / "doc_acts.npz"
    write_doc_acts(path, np.array(["doc-1", "doc-0"]))
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="value and order"):
        h1_h2_measurements.load_p1a_doc_acts(
            path, np.array(["doc-0", "doc-1"]), 4, 22, expected_sha256
        )


def test_p1a_rejects_doc_acts_from_a_different_layer(tmp_path):
    path = tmp_path / "doc_acts.npz"
    write_doc_acts(path, np.array(["doc-0", "doc-1"]), layer=12)
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="does not match basis layer"):
        h1_h2_measurements.load_p1a_doc_acts(
            path, np.array(["doc-0", "doc-1"]), 4, 22, expected_sha256
        )


def test_p1a_rejects_doc_acts_with_unexpected_sha256(tmp_path):
    path = tmp_path / "doc_acts.npz"
    write_doc_acts(path, np.array(["doc-0", "doc-1"]))

    with pytest.raises(ValueError, match="does not match the required expected digest"):
        h1_h2_measurements.load_p1a_doc_acts(path, np.array(["doc-0", "doc-1"]), 4, 22, "0" * 64)


def test_p1a_rejects_doc_acts_changed_while_loading(tmp_path, monkeypatch):
    path = tmp_path / "doc_acts.npz"
    write_doc_acts(path, np.array(["doc-0", "doc-1"]))
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    real_hash = h1_h2_measurements._sha256_hexdigest_from_handle
    calls = 0

    def changing_hash(handle):
        nonlocal calls
        calls += 1
        return real_hash(handle) if calls == 1 else "f" * 64

    monkeypatch.setattr(h1_h2_measurements, "_sha256_hexdigest_from_handle", changing_hash)
    with pytest.raises(ValueError, match="changed while it was being loaded"):
        h1_h2_measurements.load_p1a_doc_acts(
            path, np.array(["doc-0", "doc-1"]), 4, 22, expected_sha256
        )


def test_incomplete_run_requires_noncanonical_output(tmp_path):
    canonical = tmp_path / "h1_h2_measurements.json"
    canonical.write_text("completed P1a\n")

    with pytest.raises(ValueError, match="--doc-acts is required"):
        h1_h2_measurements.resolve_output_path(tmp_path, None, None, False)
    assert canonical.read_text() == "completed P1a\n"

    with pytest.raises(ValueError, match="explicit noncanonical"):
        h1_h2_measurements.resolve_output_path(tmp_path, None, None, True)
    with pytest.raises(ValueError, match="cannot write a canonical-named report"):
        h1_h2_measurements.resolve_output_path(tmp_path, canonical, None, True)
    with pytest.raises(ValueError, match="cannot write a canonical-named report"):
        h1_h2_measurements.resolve_output_path(
            tmp_path,
            tmp_path / "nested" / canonical.name,
            None,
            True,
        )
    with pytest.raises(ValueError, match="cannot write a canonical-named report"):
        h1_h2_measurements.resolve_output_path(
            tmp_path, tmp_path / canonical.name.upper(), None, True
        )

    partial = tmp_path / "partial.json"
    output, incomplete = h1_h2_measurements.resolve_output_path(tmp_path, partial, None, True)
    assert output == partial.resolve()
    assert incomplete is True
    assert canonical.read_text() == "completed P1a\n"


def test_main_rejects_exact_canonical_clobber_attack(tmp_path, monkeypatch):
    fake_script = tmp_path / "scripts" / "13_h1_h2_measurements.py"
    fake_script.parent.mkdir(parents=True)
    fake_script.touch()
    monkeypatch.setattr(h1_h2_measurements, "__file__", str(fake_script))
    canonical_name = h1_h2_measurements.CANONICAL_REPORT_NAME
    canonical = tmp_path / "results" / "h1h2" / canonical_name
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("completed P1a\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "13_h1_h2_measurements.py",
            "--allow-incomplete-p1a",
            "--out-dir",
            str(tmp_path),
            "--output",
            str(canonical),
            "--tested-units",
            "refusal",
        ],
    )

    with pytest.raises(SystemExit):
        h1_h2_measurements.main()
    assert canonical.read_text() == "completed P1a\n"


def test_write_json_atomic_cleans_temp_file_on_replace_failure(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("completed P1a\n")

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(h1_h2_measurements.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        h1_h2_measurements.write_json_atomic(output, {"status": "pending"})
    assert output.read_text() == "completed P1a\n"
    leftovers = [path for path in tmp_path.iterdir() if path != output]
    assert leftovers == []
