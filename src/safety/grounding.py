"""Grounding rules for answers generated from retrieved official evidence."""

from collections.abc import Sequence

from src.retrieval.vector_store import SearchResult

NO_EVIDENCE_MESSAGE = (
    "I could not find verified information about that question in the "
    "available official documents. Please consult the relevant government "
    "department or cooperative office."
)


def build_grounded_prompt(question: str, results: Sequence[SearchResult]) -> str:
    """Build a strict prompt that keeps the local LLM inside retrieved evidence."""

    if not question.strip():
        raise ValueError("Question must not be empty")
    if not results:
        raise ValueError("At least one retrieved result is required")

    evidence = "\n\n".join(
        (
            f"[Source: {result.chunk.source_name}, page "
            f"{result.chunk.page_number if result.chunk.page_number else 'not applicable'}"
            f"{', URL: ' + source_url if source_url else ''}]\n"
            f"{result.chunk.text}"
        )
        for result in results
        for source_url in [getattr(result.chunk, "source_url", None)]
    )
    return (
        "You are a careful public-information assistant. Answer the question "
        "using only the evidence below. If the evidence does not answer it, "
        "say that verified information is unavailable. Do not guess scheme "
        "amounts, eligibility, dates, legal conclusions, or procedures. "
        "Include source names and page numbers in the answer.\n\n"
        f"Question:\n{question}\n\nEvidence:\n{evidence}"
    )


def no_evidence_response(results: Sequence[SearchResult]) -> str | None:
    """Return a safe response when retrieval found no usable evidence."""

    if not results:
        return NO_EVIDENCE_MESSAGE
    return None
