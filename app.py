"""Streamlit interface for the local multilingual RAG assistant."""

import streamlit as st

from config.settings import NLLB_MODEL_DIR, VECTOR_STORE_DIR
from src.conversation import build_contextual_question, is_new_topic_question
from src.generation.grounded_fallback import build_fallback_answer
from src.retrieval.embeddings import LocalEmbeddingModel
from src.retrieval.query_expansion import expand_local_query
from src.retrieval.vector_store import FaissVectorStore
from src.safety.grounding import (
    NO_EVIDENCE_MESSAGE,
    no_evidence_response,
)
from src.translation.nllb_translator import NllbTranslator

st.set_page_config(
    page_title="Cooperative Governance Assistant",
    page_icon="🌾",
    layout="centered",
)

st.title("🌾 Multilingual Cooperative Governance & Legal Assistance")
st.caption("Local-first assistant for verified government information")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(message: dict[str, str]) -> None:
    with st.chat_message(message["role"]):
        st.write(message["content"])


with st.sidebar:
    st.header("Setup status")
    st.write(f"FAISS index: {'ready' if (VECTOR_STORE_DIR / 'index.faiss').is_file() else 'missing'}")
    st.write(f"NLLB model: {'ready' if NLLB_MODEL_DIR.is_dir() else 'missing'}")
    st.write(
        "Voice input/output is planned for a later stage. "
        "The current interface accepts text."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

input_language = st.selectbox("Question language", ["English", "Hindi", "Kannada"])
output_language = st.selectbox("Answer language", ["English", "Hindi", "Kannada"])

for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("Ask about a scheme, PACS service, law, or grievance process")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        if not (VECTOR_STORE_DIR / "index.faiss").is_file():
            st.warning(NO_EVIDENCE_MESSAGE)
            st.info(
                "Add verified PDFs to data/raw/government_pdfs and run "
                "`python scripts/build_index.py` first."
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": NO_EVIDENCE_MESSAGE}
            )
        else:
            try:
                if input_language != "English" or output_language != "English":
                    translator = NllbTranslator(NLLB_MODEL_DIR)
                else:
                    translator = None
                english_question = (
                    translator.translate(question, input_language, "English")
                    if translator and input_language != "English"
                    else question
                )
                previous_questions = [
                    message["content"]
                    for message in st.session_state.messages[:-1]
                    if message["role"] == "user"
                ]
                contextual_question = build_contextual_question(
                    english_question,
                    previous_questions,
                )
                encoder = LocalEmbeddingModel()
                store = FaissVectorStore(encoder)
                store.load(VECTOR_STORE_DIR)
                retrieval_question = expand_local_query(contextual_question)
                results = store.search(
                    retrieval_question,
                    top_k=6,
                    min_score=0.25,
                )
                safe_response = no_evidence_response(results)
                if safe_response:
                    st.warning(safe_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": safe_response}
                    )
                else:
                    answer_question = (
                        english_question
                        if is_new_topic_question(english_question)
                        else contextual_question
                    )
                    english_answer = build_fallback_answer(
                        answer_question,
                        results,
                    )
                    answer_source = "verified local scheme data"

                    answer = (
                        translator.translate(
                            english_answer,
                            "English",
                            output_language,
                        )
                        if translator and output_language != "English"
                        else english_answer
                    )
                    st.write(answer)
                    st.caption(
                        f"Answer based on {answer_source}. "
                        f"Question: {input_language}; answer: {output_language}"
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    with st.expander("Verified scheme and document sources"):
                        for result in results:
                            if result.chunk.source_type == "scheme":
                                data = result.chunk.structured_data or {}
                                st.markdown(f"**{result.chunk.scheme_name}**")
                                st.write(f"**Purpose:** {data.get('purpose', '')}")
                                st.write(
                                    f"**Eligibility:** {data.get('eligibility', '')}"
                                )
                                st.write(f"**Benefits:** {data.get('benefits', '')}")
                                st.write(
                                    f"**Application:** "
                                    f"{data.get('application_process', '')}"
                                )
                                if result.chunk.source_url:
                                    st.markdown(
                                        f"[Official source]({result.chunk.source_url})"
                                    )
                            else:
                                st.write(
                                    f"{result.chunk.source_name}, page "
                                    f"{result.chunk.page_number} "
                                    f"(score {result.score:.3f})"
                                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                error_message = f"Local pipeline is not ready: {error}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
