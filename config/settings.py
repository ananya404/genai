"""Central paths and language settings for the local application."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCUMENTS_DIR = DATA_DIR / "raw" / "government_pdfs"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CURATED_DATA_DIR = DATA_DIR / "curated"
SCHEMES_PATH = CURATED_DATA_DIR / "schemes.json"
MODELS_DIR = PROJECT_ROOT / "models"
NLLB_MODEL_DIR = MODELS_DIR / "nllb"
EMBEDDING_MODEL_DIR = MODELS_DIR / "embeddings"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "faiss_index"

SUPPORTED_LANGUAGES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
}
