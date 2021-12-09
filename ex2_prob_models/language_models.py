from abc import abstractmethod, ABC
import math
from collections import defaultdict, Counter

""""""""""""""""""""""""""""""""""""""
#     Dorin Keshales    313298424
#     Eran Hirsch       302620745
""""""""""""""""""""""""""""""""""""""

# The vocabulary size |V|
V = 300000


class BaseModel(ABC):
    """
    Abstract model class
    """

    # Computing the perplexity value
    def perplexity(self, val_unique_events: Counter) -> float:
        sum_of_log_probs = 0.0

        # For every unique event in the validation set
        for event in val_unique_events.keys():
            sum_of_log_probs += math.log(self.calc_prob(event), 2.0) * val_unique_events.get(event)

        count_val_unique_events = sum(val_unique_events.values())
        value = -1 * (1 / count_val_unique_events) * sum_of_log_probs

        return math.pow(2, value)

    @abstractmethod
    def calc_prob(self, input_word: str) -> float:
        raise NotImplementedError("BaseModel should not be used directly")


class LidstoneSmoothingModel(BaseModel):

    def __init__(self, lambda_param: float, count_events: int, unique_events: Counter):
        self.unique_events = unique_events
        self.count_events = count_events
        self.lambda_param = lambda_param

    # Calling the Lidstone probability calculation function
    def calc_prob(self, input_word: str) -> float:
        return self.calc_lidstone_prob(input_word)

    def calc_lidstone_prob(self, input_word: str) -> float:
        # Returning the lidstone probability of the INPUT_WORD
        return (self.unique_events.get(input_word, 0.0) + self.lambda_param) / (
                self.count_events + (self.lambda_param * V))

    def test_probabilities_sum_to_1(self):
        sum_of_probs = 0.0

        # Sum the probabilities of seen events
        for word in self.unique_events.keys():
            sum_of_probs += self.calc_lidstone_prob(word)

        # Adding the probabilities for unseen events
        num_of_unseen_events = V - len(self.unique_events)
        sum_of_probs += num_of_unseen_events * self.calc_lidstone_prob("unseen-word")

        print(f"sum_of_probs = {sum_of_probs} ; lambda = {self.lambda_param}")


class HeldoutSmoothingModel(BaseModel):

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

        # Finding events in H and not in T
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

        inversed_counter = defaultdict(list)
        for event, count in counter.items():
            inversed_counter[count].append(event)
        return inversed_counter

    # Calling the Held-out probability calculation function
    def calc_prob(self, input_word: str) -> float:
        return self.calc_heldout_prob(input_word)

    def calc_heldout_prob(self, input_word: str) -> float:
        """
        Return the held-out probability of the INPUT_WORD
        """

        # Calculating how many times the INPUT_WORD shows up in T
        word_count = self.T_unique_events.get(input_word, 0.0)

        # Calculating the sum of the frequencies of all words in H that show exactly {word_count} times in T
        t_r = sum([self.H_unique_events.get(word, 0.0) for word in self.T_inversed_counter.get(word_count, [])])

        if word_count != 0.0:
            # Calculating the number of words that show {word_count} times in T
            N_r = len(self.T_inversed_counter.get(word_count, []))
        else:
            # Calculating the number of words that show in T (words with count > 0)
            words_that_show_in_T = [key for key, value in self.T_unique_events.items() if value > 0]

            # Calculating the number of words that don't show in T (because word_count is 0)
            N_r = V - len(words_that_show_in_T)

        # Calculating the size of H
        H_size = sum(self.H_unique_events.values())

        # Held-out probability
        return t_r / (N_r * H_size)

    def test_probabilities_sum_to_1(self):
        sum_of_probs = 0.0

        # Sum the probabilities of seen events
        for word in self.T_unique_events.keys():
            sum_of_probs += self.calc_heldout_prob(word)

        # Add the probabilities for unseen events
        num_of_unseen_events = V - len(self.T_unique_events.keys())
        sum_of_probs += num_of_unseen_events * self.calc_heldout_prob("unseen-word")

        print(f"sum_of_probs = {sum_of_probs}")
