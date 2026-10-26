"""Local Ollama command-line integration with no cloud fallback."""

import subprocess


class LocalOllamaGenerator:
    """Generate text through an Ollama model running on this machine."""

    def __init__(self, model_name: str = "llama3.2:3b") -> None:
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")
        try:
            completed = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "Ollama is not installed or is not available on PATH."
            ) from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or "Ollama returned an error."
            raise RuntimeError(detail) from error

        answer = completed.stdout.strip()
        if not answer:
            raise RuntimeError("The local Ollama model returned an empty answer.")
        return answer
