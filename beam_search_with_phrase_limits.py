from datasets import load_dataset
import argparse
from transformers_phrase_limits import (
    BannedPhrases,
    PhraseLogitsProcessor,
    generate_summaries_with_phrase_limits,
    load_model_and_tokenizer
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/bart-large-xsum")
    args = parser.parse_args()
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    xsum_test = load_dataset("xsum")["test"]
    num_beams = 4

    phrase_processor = PhraseLogitsProcessor(
        tokenizer,
        num_beams,
        BannedPhrases(
            {
                "Edinburgh",
                "Wales",
                "prison",
                "charity",
                "homeless",
                "man",
                "a",
                "More needs",
            }
        ),
    )

    summaries, metadata = generate_summaries_with_phrase_limits(
        model,
        tokenizer,
        xsum_test["document"][0:2],
        phrase_processor,
        num_beams=num_beams,
        return_beam_metadata=True,
    )

    print(summaries)
