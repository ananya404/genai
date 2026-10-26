from src.generation.grounded_fallback import build_fallback_answer
from src.ingestion.chunker import TextChunk
from src.retrieval.query_expansion import expand_local_query
from src.retrieval.vector_store import SearchResult


def test_fallback_answers_single_scheme_as_chatbot() -> None:
    chunk = TextChunk(
        "scheme:0",
        "https://pgportal.gov.in/",
        "CPGRAMS",
        0,
        "grievance",
        source_type="scheme",
        source_url="https://pgportal.gov.in/",
        scheme_name="CPGRAMS",
        structured_data={
            "purpose": "Lets citizens submit and track grievances.",
            "eligibility": "Any citizen may submit a service-delivery grievance.",
            "benefits": "Provides tracking and feedback.",
            "application_process": "File online through the official portal.",
            "scheme_name": "CPGRAMS",
            "official_source_url": "https://pgportal.gov.in/",
            "verification_date": "2026-10-24",
        },
    )

    answer = build_fallback_answer(
        "How can I file a government grievance?",
        [SearchResult(chunk, 0.8)],
    )

    assert "CPGRAMS" in answer
    assert "How to proceed" in answer
    assert "official source" in answer


def test_fallback_lists_multiple_schemes() -> None:
    chunks = [
        TextChunk(
            "scheme:0",
            "url-a",
            "Scheme A",
            0,
            "a",
            source_type="scheme",
            scheme_name="Scheme A",
        ),
        TextChunk(
            "scheme:1",
            "url-b",
            "Scheme B",
            0,
            "b",
            source_type="scheme",
            scheme_name="Scheme B",
        ),
    ]

    answer = build_fallback_answer(
        "What schemes are available?",
        [SearchResult(chunks[0], 0.8), SearchResult(chunks[1], 0.7)],
    )

    assert "Scheme A" in answer
    assert "Scheme B" in answer


def test_fallback_answers_crop_damage_question_directly() -> None:
    chunk = TextChunk(
        "scheme:0",
        "https://pmfby.gov.in/",
        "PMFBY",
        0,
        "crop insurance",
        source_type="scheme",
        source_url="https://pmfby.gov.in/",
        scheme_name="Pradhan Mantri Fasal Bima Yojana (PMFBY)",
        structured_data={
            "purpose": "Provides crop-insurance coverage against specified risks.",
            "eligibility": "Farmers growing notified crops may be covered.",
            "benefits": "Insurance protection subject to policy terms.",
            "application_process": "Apply through the official portal or participating bank.",
        },
    )

    answer = build_fallback_answer(
        "My crop is destroyed, what should I do?",
        [SearchResult(chunk, 0.9)],
    )

    assert "crop has been damaged" in answer
    assert "PMFBY" in answer
    assert "participating bank" in answer


def test_local_query_expansion_handles_informal_crop_loss() -> None:
    expanded = expand_local_query("i lost my crop")

    assert "PMFBY" in expanded
    assert "crop insurance" in expanded


def test_local_query_expansion_handles_new_scheme_request() -> None:
    expanded = expand_local_query("say some other scheme")

    assert "government schemes available for farmers" in expanded


def test_fallback_gives_detailed_eligibility_answer() -> None:
    chunk = TextChunk(
        "scheme:0",
        "https://pmkisan.gov.in/",
        "PM-KISAN",
        0,
        "scheme details",
        source_type="scheme",
        source_url="https://pmkisan.gov.in/",
        scheme_name="Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)",
        structured_data={
            "purpose": "Provides income support to eligible farmer families.",
            "eligibility": "Eligible landholding farmer families are identified under guidelines.",
            "benefits": "Provides notified income-support instalments through DBT.",
            "application_process": "Register through the PM-KISAN portal or an authorised CSC.",
            "verification_date": "2026-10-24",
        },
    )

    answer = build_fallback_answer(
        "What is PM Kisan, who is eligible, and what benefits does it provide?",
        [SearchResult(chunk, 0.9)],
    )

    assert "Who it may cover" in answer
    assert "landholding farmer families" in answer
    assert "income-support instalments" in answer


def test_fallback_includes_each_scheme_purpose_in_discovery_answer() -> None:
    chunks = [
        TextChunk(
            "scheme:0",
            "url-a",
            "Scheme A",
            0,
            "a",
            source_type="scheme",
            scheme_name="Scheme A",
            structured_data={
                "purpose": "Supports irrigation.",
                "eligibility": "Farmers may qualify.",
                "application_process": "Apply at the agriculture department.",
            },
        ),
        TextChunk(
            "scheme:1",
            "url-b",
            "Scheme B",
            0,
            "b",
            source_type="scheme",
            scheme_name="Scheme B",
            structured_data={
                "purpose": "Supports farm credit.",
                "eligibility": "Farmers may qualify.",
                "application_process": "Apply at a participating bank.",
            },
        ),
    ]

    answer = build_fallback_answer(
        "What government schemes are available?",
        [SearchResult(chunks[0], 0.8), SearchResult(chunks[1], 0.7)],
    )

    assert "Supports irrigation." in answer
    assert "Supports farm credit." in answer
    assert "tell me your need" in answer.lower()


def test_explicit_scheme_name_wins_over_first_semantic_result() -> None:
    requested = TextChunk(
        "scheme:pm",
        "https://pmkisan.gov.in/",
        "PM-KISAN",
        0,
        "income support",
        source_type="scheme",
        source_url="https://pmkisan.gov.in/",
        scheme_name="Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)",
        structured_data={
            "purpose": "Provides income support to eligible farmer families.",
            "eligibility": "Eligible landholding farmer families are identified under guidelines.",
            "benefits": "Provides notified income-support instalments through DBT.",
            "application_process": "Register through the PM-KISAN portal.",
        },
    )
    unrelated = TextChunk(
        "scheme:kcc",
        "https://www.myscheme.gov.in/",
        "KCC",
        0,
        "credit",
        source_type="scheme",
        scheme_name="Kisan Credit Card (KCC) Scheme",
        structured_data={
            "purpose": "Facilitates agricultural credit.",
            "eligibility": "Farmers may qualify.",
            "benefits": "Provides a credit facility.",
            "application_process": "Apply at a participating bank.",
        },
    )

    answer = build_fallback_answer(
        "What is PM Kisan and who is eligible?",
        [SearchResult(unrelated, 0.9), SearchResult(requested, 0.8)],
    )

    assert "PM-KISAN" in answer
    assert "income support" in answer
    assert "Kisan Credit Card" not in answer
