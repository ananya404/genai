"""Deterministic query expansion for common rural-help wording."""


def expand_local_query(question: str) -> str:
    """Add locally defined retrieval terms without calling a language model."""

    normalized = question.lower()
    terms: list[str] = [question]

    if any(
        phrase in normalized
        for phrase in (
            "lost my crop",
            "lost crop",
            "crop is lost",
            "crop destroyed",
            "crop damage",
            "crop damaged",
            "crop loss",
            "my crop",
        )
    ):
        terms.append(
            "crop damage crop loss crop insurance crop destroyed "
            "Pradhan Mantri Fasal Bima Yojana PMFBY claim"
        )

    if any(
        phrase in normalized
        for phrase in ("government complaint", "file complaint", "report problem")
    ):
        terms.append(
            "government grievance complaint CPGRAMS public service grievance"
        )

    if any(
        phrase in normalized
        for phrase in ("loan for farming", "farm loan", "money for farming")
    ):
        terms.append("Kisan Credit Card KCC agricultural credit farm loan")

    if any(
        phrase in normalized
        for phrase in (
            "another scheme",
            "other scheme",
            "different scheme",
            "more schemes",
            "other schemes",
            "different schemes",
            "show schemes",
            "list schemes",
        )
    ):
        terms.append(
            "government schemes available for farmers agriculture rural citizens "
            "support benefits eligibility application"
        )

    return " ".join(terms)
