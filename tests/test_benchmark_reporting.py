"""Current reports must reject measurements from another version or source."""

import hashlib
import json
from pathlib import Path
import sys

import pytest
import jaxbind_gpu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_benchmarks as reporting  # noqa: E402


def test_report_rejects_a_different_version(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "RESULTS", tmp_path)
    (tmp_path / "current-dispatch.json").write_text(
        json.dumps({"package_version": "0.0.0"})
    )
    with pytest.raises(ValueError, match="must be measured with version"):
        reporting.load("dispatch")


def test_report_checks_application_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "RESULTS", tmp_path)
    source = tmp_path / "application.py"
    source.write_text("measured implementation\n")
    (tmp_path / "current-dispatch.json").write_text(
        json.dumps(
            {
                "package_version": jaxbind_gpu.__version__,
                "source_sha256": {
                    str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                },
            }
        )
    )
    reporting.load("dispatch")
    source.write_text("different implementation\n")
    with pytest.raises(ValueError, match="different"):
        reporting.load("dispatch")
    source.unlink()
    with pytest.raises(ValueError, match="missing measured source"):
        reporting.load("dispatch")
