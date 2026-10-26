"""Offline NLLB-200 translation wrapper."""

from pathlib import Path

from config.settings import SUPPORTED_LANGUAGES


class NllbTranslator:
    """Translate between English, Hindi, and Kannada using local weights."""

    def __init__(self, model_directory: Path) -> None:
        if not model_directory.is_dir():
            raise FileNotFoundError(
                f"NLLB model directory is missing: {model_directory}. "
                "Download the model during setup before starting the app."
            )

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_directory,
            local_files_only=True,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_directory,
            local_files_only=True,
        )

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        """Translate text using only already-downloaded local model files."""

        if not text.strip():
            raise ValueError("Text to translate must not be empty")
        if source_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported source language: {source_language}")
        if target_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported target language: {target_language}")
        if source_language == target_language:
            return text

        source_code = SUPPORTED_LANGUAGES[source_language]
        target_code = SUPPORTED_LANGUAGES[target_language]
        self._tokenizer.src_lang = source_code
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True)
        generated = self._model.generate(
            **encoded,
            forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(target_code),
            max_new_tokens=512,
        )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
