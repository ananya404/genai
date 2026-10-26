# Multilingual Cooperative Governance & Legal Assistance Chatbot

A local-first Streamlit application for Indian farmers, cooperative society members,
and rural citizens. The application is being built incrementally:

1. Curated official documents and metadata
2. Local PDF extraction and chunking
3. Local multilingual embeddings and FAISS retrieval
4. Local NLLB-200 translation
5. Grounded deterministic answer generation from verified local records
6. Streamlit chat interface
7. Optional local speech input and output

No cloud LLM API, hosted inference API, paid API, or API key is required by the
application.

## Initial setup

From the project directory in a VS Code PowerShell terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

If PowerShell blocks activation, run this once for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The large model dependencies and model weights will be added only after this
baseline environment passes its smoke test.

## One-time model setup

After the Python dependencies are installed, download the embedding and NLLB
weights while connected to the internet:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
```

This stores both checkpoints under `models/`. The application loads them with
local-only mode and does not contact Hugging Face while serving questions.

## Current local pipeline

Place only verified government PDFs in `data/raw/government_pdfs`. Preserve the
original filename because it appears in citations. The curated scheme records
are stored in `data/curated/schemes.json`, with these required fields:

- scheme name
- purpose
- eligibility
- benefits
- application process
- official source URL
- verification date
- `officially_verified: true`

The index builder combines those records with any PDF chunks:

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py
```

The command is repeatable. It rewrites
`vector_store/faiss_index/index.faiss` and `metadata.json` from the current
curated records and PDFs.

Start the Streamlit interface with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application intentionally shows a safe setup message until the local NLLB
weights and FAISS index are available. It does not download anything while
handling a user question. The active answer path does not require Ollama or
any LLM API; it composes explanations from the verified fields in each
retrieved record.

## Evidence policy

Only use documents from official government or cooperative sources. Record the
source URL separately when collecting a PDF, and do not add guessed scheme
amounts, eligibility rules, dates, or legal interpretations. Scheme records are
rejected by the loader unless they explicitly have `officially_verified: true`.
Answers cite the scheme source URL or the PDF filename and page number returned
by retrieval. A relevance threshold produces a safe no-information message when
the verified index does not contain a relevant answer.
