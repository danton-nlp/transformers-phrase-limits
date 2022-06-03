from collections import defaultdict
import torch
from transformers import LogitsProcessor
from src.beam_validators import BeamValidator


SPLIT_WORD_TOKENS = {" ", ".", ",", "_", "?", "!", "'"}


def should_backtrack(subword: str):
    return not subword.startswith(" ")


def is_subword_ending(subword: str):
    return subword.startswith(" ") or subword.endswith(" ")


class PhraseLogitsProcessor(LogitsProcessor):
    r"""
    [`PhraseLogitsProcessor`] validates phrases during beam search

    Args:
        tokenizer (`AutoTokenizer`):
            The model's tokenizer
        num_beams (`int`):
            Number of beams.
        beam_validator (`BeamValidator`):
            Checks whether a beam is valid at the phrase-level
    """

    def __init__(self, tokenizer, num_beams, beam_validator: BeamValidator):
        self.tokenizer = tokenizer
        self.beam_validator = beam_validator
        self.num_beams = num_beams
        self.excluded_beams_by_input_idx = defaultdict(list)
        self.words_to_check_by_input_idx = defaultdict(lambda: 0)
        self.failed_sequences = set()

    def is_valid_beam(
        self,
        input_idx,  # input idx being processed
        sequence,  # sequence generated so far
        token_id,  # next token to be generated (argmax of beam_scores)
        beam_scores,  # probability of all tokens to be generated
    ):
        """
        Check whether beam is valid according to the passed validators.

        To enable validating on a word-level, this method backtracks
        to collect the predicted word when it detects that the predicted
        subword (token) is a word ending.
        """
        # Token-level checks
        # if token_id in self.banned_token_ids:
        #     return False

        # Word-level checks
        current_subword = self.tokenizer.decode(token_id)
        backtrack_word = ""
        is_subword_ending = False
        for char in current_subword:
            if char in SPLIT_WORD_TOKENS:
                is_subword_ending = True
                break
            else:
                backtrack_word += char

        # if the predicted subword indicates a word ending
        # backtrack to collect the predicted word
        backtrack_done = False
        if is_subword_ending:
            prev_subword_idx = len(sequence) - 1
            while prev_subword_idx != 0 and not backtrack_done:
                prev_token_id = sequence[prev_subword_idx]
                prev_subword = self.tokenizer.decode(prev_token_id)
                prev_char_idx = len(prev_subword) - 1
                while prev_char_idx >= 0:
                    prev_char = prev_subword[prev_char_idx]
                    if prev_char not in SPLIT_WORD_TOKENS:
                        backtrack_word = prev_char + backtrack_word
                    else:
                        if self.beam_validator.is_maybe_invalid_phrase_ending(
                            prev_char + backtrack_word, input_idx
                        ):
                            backtrack_word = prev_char + backtrack_word
                        else:
                            backtrack_done = True
                            break
                    prev_char_idx -= 1
                prev_subword_idx -= 1
            self.words_to_check_by_input_idx[input_idx] += 1
            if not self.beam_validator.is_valid_beam(
                backtrack_word, input_idx, sequence, beam_scores
            ):
                return False
        return True

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        blocked_beams_by_input_idx = defaultdict(lambda: 0)
        # for every beam (partially generated sentence)
        for beam_idx, (beam_input_ids, beam_scores) in enumerate(
            zip(input_ids, scores)
        ):
            top_k = beam_scores.topk(k=1)
            for prob, idx in zip(top_k[0], top_k[1]):
                input_idx = beam_idx // self.num_beams
                if not self.is_valid_beam(
                    input_idx, beam_input_ids, idx.item(), scores[beam_idx]
                ):
                    scores[beam_idx, :] = -float("inf")
                    self.excluded_beams_by_input_idx[input_idx].append(
                        (
                            beam_input_ids,
                            idx.item(),
                            prob.item(),
                        )
                    )
                    blocked_beams_by_input_idx[input_idx] += 1

        for input_idx, n_blocked in blocked_beams_by_input_idx.items():
            if n_blocked == self.num_beams:
                self.failed_sequences.add(input_idx)

        return scores
