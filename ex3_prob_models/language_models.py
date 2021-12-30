""""""""""""""""""""""""""""""""""""""
#     Dorin Keshales    313298424
#     Eran Hirsch       302620745
""""""""""""""""""""""""""""""""""""""


import math
from abc import ABC, abstractmethod
from collections import Counter

# The vocabulary size |V|
V = 300000


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

    def __init__(self, lambda_param: float, count_events: int, unique_events: Counter):
        self.unique_events = unique_events
        self.count_events = count_events
        self.lambda_param = lambda_param

    def calc_prob(self, input_word: str) -> float:
        """
        Returns the lidstone probability of the input_word
        """

        return self.calc_prob_by_word_count(self.unique_events.get(input_word, 0.0))

    def calc_prob_by_word_count(self, word_count: int) -> float:
        """
        Returns the lidstone probability of the input word_count (separated for output 29)
        """

        return (word_count + self.lambda_param) / (self.count_events + (self.lambda_param * V))

    def calc_f_lambda(self, r: int) -> float:
        """
        Returns the f_λ value required for output 29
        """

        # Calculating the lidstone probability of the word_count r
        prob_by_word_count = self.calc_prob_by_word_count(r)

        # Rounding the f_λ value to 5 digits after the decimal point
        return round(prob_by_word_count * self.count_events, 5)

    def test_probabilities_sum_to_1(self):

        sum_of_probs = 0.0

        # Sum the probabilities of seen events
        for word in self.unique_events.keys():
            sum_of_probs += self.calc_prob(word)

        # Adding the probabilities for unseen events
        num_of_unseen_events = V - len(self.unique_events)
        sum_of_probs += num_of_unseen_events * self.calc_prob("unseen-word")

        print(f"sum_of_probs = {sum_of_probs} ; lambda = {self.lambda_param}")
