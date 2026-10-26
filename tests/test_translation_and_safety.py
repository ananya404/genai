from src.safety.grounding import (
    NO_EVIDENCE_MESSAGE,
    build_grounded_prompt,
    no_evidence_response,
)


def test_missing_evidence_returns_safe_message() -> None:
    assert no_evidence_response([]) == NO_EVIDENCE_MESSAGE


def test_grounded_prompt_contains_citation_metadata() -> None:
    class Chunk:
        source_name = "official.pdf"
        page_number = 4
        text = "The official procedure is described here."

    class Result:
        chunk = Chunk()

    prompt = build_grounded_prompt("How do I apply?", [Result()])

    assert "official.pdf" in prompt
    assert "page 4" in prompt
    assert "Do not guess" in prompt
