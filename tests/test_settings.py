from config.settings import SUPPORTED_LANGUAGES


def test_supported_languages_include_required_languages() -> None:
    assert SUPPORTED_LANGUAGES == {
        "English": "eng_Latn",
        "Hindi": "hin_Deva",
        "Kannada": "kan_Knda",
    }
