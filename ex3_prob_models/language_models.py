import math
from abc import ABC, abstractmethod
from collections import Counter

""""""""""""""""""""""""""""""""""""""
#     Dorin Keshales    313298424
#     Eran Hirsch       302620745
""""""""""""""""""""""""""""""""""""""


class BaseSmoothingModel(ABC):
    """
    Abstract model class
    """

    def perplexity(self, val_unique_events: Counter) -> float:
        """
        Computing the perplexity value
        """

        sum_of_log_probs = 0.0

        # For every unique event in the validation set
        for event in val_unique_events.keys():
            sum_of_log_probs += math.log(self.calc_prob(event), 2.0) * val_unique_events.get(event)

        count_val_unique_events = sum(val_unique_events.values())
        value = -1 * (1 / count_val_unique_events) * sum_of_log_probs

        # Returning the perplexity value
        return math.pow(2, value)

    @abstractmethod
    def calc_prob(self, input_word: str) -> float:
        raise NotImplementedError("BaseModel should not be used directly")


class LidstoneSmoothingModel(BaseSmoothingModel):
    """
    Lidstone smoothing model class
    """

    def __init__(self, lambda_param: float, count_events: int, unique_events: Counter, vocab_size: int):
        self.unique_events = unique_events
        self.count_events = count_events
        self.lambda_param = lambda_param
        self.vocab_size = vocab_size

    def calc_prob(self, input_word: str) -> float:
        """
        Returns the lidstone probability of the input_word
        """

        return self.calc_prob_by_word_count(self.unique_events.get(input_word, 0.0))

    def calc_prob_by_word_count(self, word_count: int) -> float:
        """
        Returns the lidstone probability of the input word_count
        """

        return (word_count + self.lambda_param) / (self.count_events + (self.lambda_param * self.vocab_size))

    def test_probabilities_sum_to_1(self):
        sum_of_probs = 0.0

        # Sum the probabilities of seen events
        for word in self.unique_events.keys():
            sum_of_probs += self.calc_prob(word)

        # Adding the probabilities for unseen events
        num_of_unseen_events = self.vocab_size - len(self.unique_events)
        sum_of_probs += num_of_unseen_events * self.calc_prob("unseen-word")

        assert abs(sum_of_probs - 1.0) < 0.005
