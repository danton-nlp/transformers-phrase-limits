from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict


class BeamValidator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def is_valid_beam(self, current_phrase, input_idx, beam_sequence, beam_scores):
        """
        TODO
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def is_maybe_invalid_phrase_ending(self, ending, input_idx):
        """
        TODO
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class BannedPhrases(BeamValidator):
    def __init__(
        self, banned_phrases=set(), banned_phrases_by_input_idx: Dict[int, set] = {}
    ):
        self.banned_phrases_by_idx = defaultdict(
            lambda: banned_phrases, banned_phrases_by_input_idx
        )

    def is_valid_beam(self, current_phrase, input_idx, beam_sequence, beam_scores):
        return current_phrase not in self.banned_phrases_by_idx[input_idx]

    def is_maybe_invalid_phrase_ending(self, ending, input_idx):
        for phrase in self.banned_phrases_by_idx[input_idx]:
            if phrase.endswith(ending):
                return True


class EnforceSourceOverlap(BeamValidator):
    def __init__(self, docs_to_summarize):
        self.docs_to_summarize = docs_to_summarize

    def is_valid_beam(self, current_phrase, input_idx, beam_sequence, beam_scores):
        return current_phrase.lower() in self.docs_to_summarize[input_idx]
