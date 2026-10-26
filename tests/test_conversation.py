import pytest

from src.conversation import build_contextual_question, is_new_topic_question


def test_contextual_question_includes_recent_user_questions() -> None:
    contextual = build_contextual_question(
        "What should I do?",
        ["I lost my crop"],
    )

    assert "I lost my crop" in contextual
    assert "What should I do?" in contextual


def test_contextual_question_limits_history() -> None:
    contextual = build_contextual_question(
        "Current",
        ["old", "middle", "recent"],
        max_previous_questions=2,
    )

    assert "old" not in contextual
    assert "middle" in contextual
    assert "recent" in contextual


def test_contextual_question_rejects_empty_question() -> None:
    with pytest.raises(ValueError):
        build_contextual_question("", [])


def test_explicit_new_scheme_topic_does_not_use_old_context() -> None:
    contextual = build_contextual_question(
        "say some other scheme",
        ["I lost my crop"],
    )

    assert "I lost my crop" not in contextual


def test_new_topic_detection_handles_common_wording() -> None:
    assert is_new_topic_question("Tell me a different scheme")
    assert is_new_topic_question("Show me some other schemes")
    assert not is_new_topic_question("What should I do?")
