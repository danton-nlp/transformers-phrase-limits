from datasets import load_dataset
import pytest
from src.phrase_logits_processor import PhraseLogitsProcessor
from src.beam_validators import BannedPhrases
from src.generation_utils import (
    generate_summaries_with_phrase_limits,
    load_model_and_tokenizer,
)


@pytest.fixture(scope="session")
def bart_xsum():
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("facebook/bart-large-xsum")
    return model, tokenizer


@pytest.fixture(scope="module")
def docs_to_summarize():
    xsum_test = load_dataset("xsum")["test"]
    return xsum_test["document"][0:1]


def test_no_constraints(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum
    num_beams = 4

    summary = generate_summaries_with_phrase_limits(
        model, tokenizer, docs_to_summarize, None, num_beams
    )[0]

    assert "Wales" in summary
    assert "former prison" in summary


def test_banned_word(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum
    num_beams = 4

    phrase_processor = PhraseLogitsProcessor(
        tokenizer,
        num_beams,
        BannedPhrases(
            {
                "Wales",
            }
        ),
    )

    summary = generate_summaries_with_phrase_limits(
        model, tokenizer, docs_to_summarize, phrase_processor, num_beams
    )[0]

    assert "Wales" not in summary


def test_banned_phrase(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum
    num_beams = 4

    phrase_processor = PhraseLogitsProcessor(
        tokenizer,
        num_beams,
        BannedPhrases({"former prison"}),
    )

    summary = generate_summaries_with_phrase_limits(
        model, tokenizer, docs_to_summarize, phrase_processor, num_beams
    )[0]

    assert "former prison" not in summary


def test_failed_generation_one_beam(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum
    num_beams = 1

    phrase_processor = PhraseLogitsProcessor(
        tokenizer,
        num_beams,
        BannedPhrases({"prison"}),
    )

    summary = generate_summaries_with_phrase_limits(
        model, tokenizer, docs_to_summarize, phrase_processor, num_beams
    )[0]

    assert summary == "<Failed generation: blocked all beams>"
    assert 0 in phrase_processor.failed_sequences


def test_failed_generation_multiple_beams(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum
    num_beams = 4

    phrase_processor = PhraseLogitsProcessor(
        tokenizer,
        num_beams,
        BannedPhrases(
            {"Wales", "prison", "accommodation", "charity", "housing", "former", "more"}
        ),
    )

    summary = generate_summaries_with_phrase_limits(
        model, tokenizer, docs_to_summarize, phrase_processor, num_beams
    )[0]

    assert summary == "<Failed generation: blocked all beams>"
    assert 0 in phrase_processor.failed_sequences
