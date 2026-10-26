import json
from pathlib import Path

import pytest

from src.ingestion.scheme_loader import load_scheme_records


def valid_record() -> dict[str, object]:
    return {
        "scheme_name": "Test scheme",
        "purpose": "Test purpose",
        "eligibility": "Test eligibility",
        "benefits": "Test benefits",
        "application_process": "Test application",
        "official_source_url": "https://example.gov.in/scheme",
        "verification_date": "2026-10-23",
        "officially_verified": True,
    }


def test_load_scheme_records_creates_citation_metadata(tmp_path: Path) -> None:
    path = tmp_path / "schemes.json"
    path.write_text(json.dumps([valid_record()]), encoding="utf-8")

    chunks = load_scheme_records(path)

    assert len(chunks) == 1
    assert chunks[0].source_type == "scheme"
    assert chunks[0].scheme_name == "Test scheme"
    assert chunks[0].source_url == "https://example.gov.in/scheme"
    assert chunks[0].structured_data["purpose"] == "Test purpose"


def test_unverified_record_is_rejected(tmp_path: Path) -> None:
    record = valid_record()
    record["officially_verified"] = False
    path = tmp_path / "schemes.json"
    path.write_text(json.dumps([record]), encoding="utf-8")

    with pytest.raises(ValueError, match="officially_verified"):
        load_scheme_records(path)
