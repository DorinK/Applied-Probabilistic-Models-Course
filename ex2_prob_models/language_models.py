from abc import abstractmethod, ABC
import math
from collections import defaultdict, Counter
from typing import Tuple


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


class HeldoutSmoothingModel(BaseSmoothingModel):
    """
    Held-out smoothing model class
    """

    def __init__(self, T_unique_events: Counter, H_unique_events: Counter):
        new_T_unique_events = self.add_events_to_T_based_on_H(T_unique_events, H_unique_events)
        self.T_unique_events = new_T_unique_events
        self.T_inversed_counter = self.inverse_counter(new_T_unique_events)
        self.H_unique_events = H_unique_events

    @staticmethod
    def add_events_to_T_based_on_H(T_unique_events: Counter, H_unique_events: Counter) -> Counter:
        """
        For held-out, we need to easily find events in H that are not in T, so we add these events to the T counter as 0.0
        """

        # Creating a copy to avoid changing a variable received by reference
        new_T_unique_events = T_unique_events.copy()

        # Finding events in H that are not in T
        events_in_H_not_in_T = [event for event in H_unique_events.keys() if event not in new_T_unique_events]

        # Adding these events to T with count 0.0
        for event_in_H_not_in_T in events_in_H_not_in_T:
            new_T_unique_events[event_in_H_not_in_T] = 0.0

        return new_T_unique_events

    @staticmethod
    def inverse_counter(counter: Counter) -> dict:
        """
        Inverses a counter. For example, {"a": 5, "b": 5, "c": 6} will turn to {5: ["a", "b"], 6: ["c"]}
        """

        inverted_counter = defaultdict(list)

        # For each pair of event and count, make the count a key and the event a value of this key
        for event, count in counter.items():
            inverted_counter[count].append(event)

        # Returning the inverted counter
        return inverted_counter

    def calc_prob(self, input_word: str) -> float:
        """
        Returns the held-out probability of the input_word
        """

        # Calculating how many times the input_word shows up in T
        word_count = self.T_unique_events.get(input_word, 0.0)

        # Calculating the t_r and N_r values
        t_r = self._calc_t_r(word_count)
        N_r = self._calc_N_r(word_count)

        # Calculating the size of H
        H_size = sum(self.H_unique_events.values())

        # Returning the held-out probability
        return t_r / (N_r * H_size)

    def _calc_t_r(self, r: int) -> int:
        """
        Returns the sum of the frequencies of all words in H that show exactly {r} times in T
        Returns the value of t_r
        """

        # Returning the value of t_r
        return sum([self.H_unique_events.get(word, 0.0) for word in self.T_inversed_counter.get(r, [])])

    def _calc_N_r(self, r: int) -> int:
        """
        Returns the number of events of frequency r in the training half of the development set, which is used in the
        held-out estimation
        """

        if r != 0:
            # Calculating the number of words that show {word_count} times in T
            N_r = len(self.T_inversed_counter.get(r, []))
        else:
            # Calculating the number of words that show in T (words with count > 0)
            words_that_show_in_T = [key for key, value in self.T_unique_events.items() if value > 0]

            # Calculating the number of words that don't show in T (because word_count is 0)
            N_r = V - len(words_that_show_in_T)

        # Returning the value of N_r
        return N_r

    def calc_values_for_output_29(self, r: int) -> Tuple[float, int, int]:
        """
        Returns the values required for output 29
        """

        # Calculating the t_r and N_r values
        t_r = self._calc_t_r(r)
        N_r = self._calc_N_r(r)

        # Calculating the expected frequency according to the estimated p(x) on the same corpus from which the original
        # r was counted
        f_H = t_r / N_r

        # Returning the values of f_H, N_r and t_r (rounding the result of f_H to 5 digits after the decimal point)
        return round(f_H, 5), N_r, int(t_r)

    def test_probabilities_sum_to_1(self):

        sum_of_probs = 0.0

        # Sum the probabilities of seen events
        for word in self.T_unique_events.keys():
            sum_of_probs += self.calc_prob(word)

        # Add the probabilities for unseen events
        num_of_unseen_events = V - len(self.T_unique_events.keys())
        sum_of_probs += num_of_unseen_events * self.calc_prob("unseen-word")

        print(f"sum_of_probs = {sum_of_probs}")
