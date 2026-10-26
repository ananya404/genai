"""Load verified, curated scheme records for local RAG indexing."""

import json
from pathlib import Path
from typing import Any

from src.ingestion.chunker import TextChunk

REQUIRED_FIELDS = (
    "scheme_name",
    "purpose",
    "eligibility",
    "benefits",
    "application_process",
    "official_source_url",
    "verification_date",
    "officially_verified",
)


def load_scheme_records(path: Path) -> list[TextChunk]:
    """Load verified scheme records and convert them to searchable chunks."""

    if not path.is_file():
        raise FileNotFoundError(f"Scheme database does not exist: {path}")
    raw_records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise ValueError("Scheme database must contain a JSON list")

    chunks: list[TextChunk] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise ValueError(f"Scheme record {index} must be a JSON object")
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Scheme record {index} is missing: {', '.join(missing)}")
        if record["officially_verified"] is not True:
            raise ValueError(
                f"Scheme record {index} is not marked officially_verified=true"
            )
        for field in REQUIRED_FIELDS[:-1]:
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(
                    f"Scheme record {index} field {field} must be a non-empty string"
                )

        searchable_text = _record_to_search_text(record)
        chunks.append(
            TextChunk(
                chunk_id=f"scheme:{index}:{record['scheme_name']}",
                source_path=record["official_source_url"],
                source_name=record["scheme_name"],
                page_number=0,
                text=searchable_text,
                source_type="scheme",
                source_url=record["official_source_url"],
                scheme_name=record["scheme_name"],
                structured_data={
                    field: str(record[field])
                    for field in REQUIRED_FIELDS
                    if field != "officially_verified"
                },
            )
        )
    return chunks


def _record_to_search_text(record: dict[str, Any]) -> str:
    return "\n".join(
        (
            f"Scheme name: {record['scheme_name']}",
            f"Purpose: {record['purpose']}",
            f"Eligibility: {record['eligibility']}",
            f"Benefits: {record['benefits']}",
            f"Application process: {record['application_process']}",
            f"Official source: {record['official_source_url']}",
            f"Verified on: {record['verification_date']}",
        )
    )
