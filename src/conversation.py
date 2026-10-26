"""Helpers for bounded, local conversation context."""

from collections.abc import Sequence


def is_new_topic_question(question: str) -> bool:
    """Return whether the user explicitly asks to change the conversation topic."""

    normalized = question.lower().strip()
    return any(
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
    )


def build_contextual_question(
    question: str,
    previous_questions: Sequence[str],
    max_previous_questions: int = 3,
) -> str:
    """Combine the current question with recent user questions for retrieval."""

    if not question.strip():
        raise ValueError("Question must not be empty")
    if max_previous_questions < 0:
        raise ValueError("max_previous_questions must not be negative")
    if is_new_topic_question(question):
        return question

    recent = [item.strip() for item in previous_questions if item.strip()][
        -max_previous_questions:
    ]
    if not recent:
        return question
    return "Previous user context: " + " | ".join(recent) + f"\nCurrent question: {question}"
