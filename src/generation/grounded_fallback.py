"""Deterministic, local fallback answers from retrieved verified records."""

from collections.abc import Sequence

from src.retrieval.vector_store import SearchResult


def _contains_any(question: str, terms: tuple[str, ...]) -> bool:
    return any(term in question for term in terms)


def _best_scheme(
    results: Sequence[SearchResult],
    names: tuple[str, ...],
) -> SearchResult | None:
    for result in results:
        scheme_name = (result.chunk.scheme_name or "").lower()
        if any(name in scheme_name for name in names):
            return result
    return None


def _scheme_name(result: SearchResult) -> str:
    return result.chunk.scheme_name or result.chunk.source_name


def _scheme_data(result: SearchResult) -> dict[str, str]:
    return result.chunk.structured_data or {}


def _explicit_scheme_match(
    question: str,
    results: Sequence[SearchResult],
) -> SearchResult | None:
    normalized = question.lower().replace("-", " ")
    aliases = {
        "pm kisan": "pradhan mantri kisan samman nidhi",
        "pmfby": "fasal bima",
        "crop insurance": "fasal bima",
        "kcc": "kisan credit card",
        "pm kusum": "pm kusum",
        "soil health": "soil health card",
        "e nam": "national agriculture market",
        "cpgrams": "grievance",
    }
    for alias, expected in aliases.items():
        if alias in normalized:
            for result in results:
                if expected in _scheme_name(result).lower().replace("-", " "):
                    return result

    question_words = {
        word for word in normalized.split() if len(word) >= 4
    }
    best: SearchResult | None = None
    best_overlap = 0
    for result in results:
        scheme_words = {
            word
            for word in _scheme_name(result).lower().replace("-", " ").split()
            if len(word) >= 4
        }
        overlap = len(question_words & scheme_words)
        if overlap > best_overlap:
            best = result
            best_overlap = overlap
    return best


def _source_note(result: SearchResult) -> str:
    data = _scheme_data(result)
    verified = data.get("verification_date", "the recorded verification date")
    url = result.chunk.source_url or data.get("official_source_url", "")
    return (
        f"Source: {url or result.chunk.source_name}. "
        f"Record verified on {verified}. Check the official source for current "
        "state, season, deadline, and eligibility conditions."
    )


def _detailed_scheme_answer(
    result: SearchResult,
    question: str,
    focus: str = "overview",
) -> str:
    name = _scheme_name(result)
    data = _scheme_data(result)
    purpose = data.get("purpose", "The available record does not describe the purpose.")
    eligibility = data.get(
        "eligibility",
        "The available record does not provide enough eligibility detail.",
    )
    benefits = data.get(
        "benefits",
        "The available record does not provide detailed benefit information.",
    )
    application = data.get(
        "application_process",
        "Please use the official source to confirm the application process.",
    )

    if focus == "eligibility":
        opening = f"{name} is relevant because {purpose}"
        body = (
            f"Who may qualify: {eligibility}\n\n"
            "This scheme is intended to address the need described above. "
            "This is general scheme information, not a final eligibility decision. "
            "The implementing department or lender may check land records, identity, "
            "location, season, documents, or other conditions."
        )
    elif focus == "benefits":
        opening = f"Here is what {name} may provide."
        body = (
            f"{benefits}\n\n"
            "The record does not guarantee that every applicant will receive the "
            "same support. The actual assistance depends on the applicable guidelines, "
            "approval, available component, and implementing authority."
        )
    elif focus == "application":
        opening = f"To proceed with {name}, start with the following verified guidance."
        body = (
            f"{application}\n\n"
            f"Who it may cover: {eligibility}\n\n"
            "Before submitting anything, confirm the current form, deadline, "
            "required documents, and local office or portal shown by the official "
            "source. Keep the acknowledgement or application number for follow-up."
        )
    elif focus == "documents":
        opening = f"For {name}, the available record does not list a fixed document checklist."
        body = (
            f"Start by checking the official process: {application}\n\n"
            "Depending on the scheme and your situation, the implementing authority "
            "may verify identity, contact details, bank information, land or crop "
            "records, and other scheme-specific information. Do not rely on an "
            "unofficial checklist; confirm the exact requirements before applying."
        )
    else:
        opening = f"{name} is relevant because {purpose}"
        body = (
            f"Who it may cover: {eligibility}\n\n"
            f"What it may provide: {benefits}\n\n"
            f"How to proceed: {application}"
        )

    return f"{opening}\n\n{body}\n\n{_source_note(result)}"


def _answer_for_intent(
    question: str,
    results: Sequence[SearchResult],
) -> str | None:
    normalized = question.lower()
    crop_damage = _contains_any(
        normalized,
        (
            "crop destroyed",
            "crop is destroyed",
            "my crop is destroyed",
            "crop damage",
            "crop damaged",
            "lost my crop",
            "crop loss",
            "flood damage",
            "drought damage",
            "hail damage",
        ),
    )
    if crop_damage:
        pmfby = _best_scheme(results, ("fasal bima", "pmfby"))
        if pmfby:
            return (
                "I'm sorry to hear that your crop has been damaged. The most "
                "relevant verified record is below.\n\n"
                + _detailed_scheme_answer(pmfby, question, "application")
                + "\n\n"
                "Check the current season and State or Union Territory "
                "notification for the applicable crop, area, reporting, and "
                "claim procedure. Do not rely on an old deadline or assume "
                "that a claim is automatically payable.\n\n"
                "If you are already insured, contact your participating bank "
                "or PACS, insurer, or local agriculture department. Keep your "
                "policy, crop records, and any loss evidence available."
            )

    result = _explicit_scheme_match(question, results) or results[0]
    asks_about_scheme = _contains_any(
        normalized,
        ("what is", "tell me about", "explain", "how does", "overview"),
    )
    asks_about_eligibility = _contains_any(
        normalized,
        ("am i eligible", "who is eligible", "eligibility", "can i get"),
    )
    asks_about_benefits = _contains_any(
        normalized,
        ("what does it provide", "what are the benefits", "benefit", "support", "how much"),
    )
    if asks_about_scheme and (asks_about_eligibility or asks_about_benefits):
        return _detailed_scheme_answer(result, question)

    if _contains_any(
        normalized,
        ("how do i apply", "how can i apply", "where can i apply", "apply for", "register"),
    ):
        return _detailed_scheme_answer(result, question, "application")

    if _contains_any(
        normalized,
        ("am i eligible", "who is eligible", "eligibility", "can i get"),
    ):
        return _detailed_scheme_answer(result, question, "eligibility")

    if _contains_any(
        normalized,
        ("what does it provide", "what are the benefits", "benefit", "support", "how much"),
    ):
        return _detailed_scheme_answer(result, question, "benefits")

    if _contains_any(
        normalized,
        ("document", "documents", "paper", "papers", "what do i need", "status"),
    ):
        return _detailed_scheme_answer(result, question, "documents")

    if _contains_any(
        normalized,
        ("grievance", "complaint", "complain", "report a problem"),
    ):
        cpgrams = _best_scheme(results, ("cpgrams", "grievance"))
        if cpgrams:
            data = cpgrams.chunk.structured_data or {}
            return (
                "You can use the Centralised Public Grievance Redress and "
                "Monitoring System (CPGRAMS) for a public-service grievance.\n\n"
                f"How to proceed: {data.get('application_process', '')}\n\n"
                f"{data.get('benefits', '')}\n\n"
                f"{_source_note(cpgrams)}"
            )
    return None


def build_fallback_answer(
    question: str,
    results: Sequence[SearchResult],
) -> str:
    """Create a concise chatbot-style answer without an LLM or network call."""

    if not question.strip():
        raise ValueError("Question must not be empty")
    if not results:
        raise ValueError("At least one retrieved result is required")

    scheme_results = [
        result for result in results if result.chunk.source_type == "scheme"
    ]
    if scheme_results:
        explicit_match = _explicit_scheme_match(question, scheme_results)
        if explicit_match:
            intent_answer = _answer_for_intent(question, [explicit_match])
            return intent_answer or _detailed_scheme_answer(
                explicit_match,
                question,
            )
        intent_answer = _answer_for_intent(question, scheme_results)
        if intent_answer:
            return intent_answer
        if len(scheme_results) == 1:
            return _detailed_scheme_answer(scheme_results[0], question)

        summaries = []
        for result in scheme_results:
            data = _scheme_data(result)
            summaries.append(
                f"**{_scheme_name(result)}**\n"
                f"Purpose: {data.get('purpose', 'Not stated in the record.')}\n"
                f"Who it may cover: {data.get('eligibility', 'Check the official rules.')}\n"
                f"How to start: {data.get('application_process', 'Check the official source.')}"
            )
        return (
            "I found several verified government schemes that may be relevant. "
            "They serve different needs, so the best option depends on your crop, "
            "location, land or work situation, and the support you need.\n\n"
            + "\n\n".join(summaries)
            + "\n\nAsk about one scheme by name, or tell me your need—for example "
            "crop insurance, farm credit, irrigation, soil testing, market access, "
            "employment, housing, pension, or a grievance—and I will narrow the "
            "information to the most relevant records."
        )

    best = results[0].chunk
    return (
        "I found relevant information in the available official documents, "
        f"including {best.source_name}, page {best.page_number}. "
        "Please ask a more specific question so I can summarize the applicable "
        "procedure from that source."
    )
