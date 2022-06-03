from datasets import load_dataset
import argparse
from src.generation_utils import generate_summaries, load_model_and_tokenizer
from src.beam_validators import BannedPhrases
from src.word_logits_processor import WordLogitsProcessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/bart-large-xsum")
    args = parser.parse_args()
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    xsum_test = load_dataset("xsum")["test"]
    num_beams = 4

    factuality_enforcer = WordLogitsProcessor(
        tokenizer, 
        num_beams, 
        BannedPhrases({
            "Edinburgh",
            "Wales",
            "prison",
            "charity",
            "homeless",
            "man",
            "a",
            "More needs"
        })
    )

    summaries, metadata = generate_summaries(
        model, 
        tokenizer, 
        xsum_test["document"][0:2], 
        factuality_enforcer,
        num_beams=num_beams,
        return_beam_metadata=True
    )

    print(summaries)
