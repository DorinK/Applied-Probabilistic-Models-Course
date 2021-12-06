import math
import sys
from abc import abstractmethod, ABC
from collections import defaultdict, Counter

""""""""""""""""""""""""""""""""""""""
#     Dorin Keshales    313298424
#     Eran Hirsch       302620745
""""""""""""""""""""""""""""""""""""""

# The vocabulary size |V|
V = 300000

# Command-line arguments
development_set_filename = sys.argv[1]
test_set_filename = sys.argv[2]
INPUT_WORD = sys.argv[3]
output_filename = sys.argv[4]


class BaseModel(ABC):
    """
    Abstract model class
    """

    def perplexity(self, val_unique_events: Counter) -> float:
        sum_of_log_probs = 0.0
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

    def calc_prob(self, input_word: str) -> float:
        return self.calc_lidstone_prob(input_word)

    def calc_lidstone_prob(self, input_word: str) -> float:
        # Returning the lidstone probability of the input word.
        return (self.unique_events.get(input_word, 0.0) + self.lambda_param) / (
                self.count_events + (self.lambda_param * V))  # TODO: Check this - Why V?

    def test_probabilities_sum_to_1(self):
        # Sum probabilities of seen events
        sum_of_probs = 0.0
        for word in self.unique_events.keys():
            sum_of_probs += self.calc_lidstone_prob(word)

        # Add probabilities for unseen events
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
        For heldout we need to easily find events in H that are not in T, so we add these events to the T counter as 0.0
        """

        # Create a copy to avoid changing a variable received by reference
        new_T_unique_events = T_unique_events.copy()

        # Find events in H and not in T
        events_in_H_not_in_T = [event for event in H_unique_events.keys() if event not in new_T_unique_events]

        # Add these events to T with count 0.0
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

    def calc_prob(self, input_word: str) -> float:
        return self.calc_heldout_prob(input_word)

    def calc_heldout_prob(self, input_word: str) -> float:
        """
        Return the heldout probability of the input word.
        """

        # Calc how many times the input_word shows up in T
        word_count = self.T_unique_events.get(input_word, 0.0)

        # Calc the sum of the frequencies of all words in H that show exactly {word_count} times in T
        t_r = sum([self.H_unique_events.get(word, 0.0) for word in self.T_inversed_counter.get(word_count, [])])

        if word_count != 0.0:
            # Calc the number of words that show {word_count} times in T
            N_r = len(self.T_inversed_counter.get(word_count, []))
        else:
            # Calc the number of words that show in T (words with count > 0)
            words_that_show_in_T = [key for key, value in self.T_unique_events.items() if value > 0]

            # Calc the number of words that don't show in T (because word_count is 0)
            N_r = V - len(words_that_show_in_T)

        # Calc the size of H
        H_size = sum(self.H_unique_events.values())

        # Heldout probability
        return t_r / (N_r * H_size)

    def test_probabilities_sum_to_1(self):
        # Sum probabilities of seen events
        sum_of_probs = 0.0
        import tqdm
        for word in tqdm.tqdm(self.T_unique_events.keys()):
            sum_of_probs += self.calc_heldout_prob(word)

        # Add probabilities for unseen events
        num_of_unseen_events = V - len(self.T_unique_events.keys())
        sum_of_probs += num_of_unseen_events * self.calc_heldout_prob("unseen-word")

        print(f"sum_of_probs = {sum_of_probs}")


def find_best_lambda_param(train_count_events: int, train_unique_events: Counter, val_unique_events: Counter):
    best_lambda = None
    best_lambda_perplexity = math.inf  # Fixed - replaced none with math.inf

    # Iterate over a range of optional lambda values
    for lambda_param_int in range(0, 200, 1):
        lambda_param = lambda_param_int / 100
        if lambda_param == 0: continue  # Ignoring λ = 0, since log is not defined on 0 value.

        # Calculate perplexity of a lidstone model
        model = LidstoneSmoothingModel(lambda_param, train_count_events, train_unique_events)
        curr_perplexity = model.perplexity(val_unique_events)

        # Check if we found a better perplexity than before
        was_better_perplexity_found = curr_perplexity < best_lambda_perplexity

        if best_lambda is None or was_better_perplexity_found:
            best_lambda = lambda_param
            best_lambda_perplexity = curr_perplexity

    return best_lambda, best_lambda_perplexity


if __name__ == '__main__':

    # Opening the output file.
    with open(output_filename, 'w', encoding='utf-8') as f:

        # Writing the students info to the output file.
        f.write('#Students\tDorin Keshales\tEran Hirsch\t313298424\t302620745\n')
        # Writing the development set file name to the output file.
        f.write(f'#Output1\t{development_set_filename}\n')
        # Writing the test set file name to the output file.
        f.write(f'#Output2\t{test_set_filename}\n')
        # Writing the input word to the output file.
        f.write(f'#Output3\t{INPUT_WORD}\n')
        # Writing the output file name to the output file.
        f.write(f'#Output4\t{output_filename}\n')

        # Writing the language vocabulary size to the output file.
        f.write(f'#Output5\t{V}\n')

        # Writing the probability of P_uniform(Event = INPUT WORD) to the output file.
        f.write(f'#Output6\t{1 / V}\n')

        # Reading the development set file and pulling out the words only in the relevant lines.
        with open(development_set_filename, 'r', encoding='utf-8') as dev:
            file = [x[:-2] if x.endswith("\n") else x for x in dev.readlines()]

        S = []
        for line in file:
            if line.startswith('<TRAIN') or not line:
                continue
            else:
                for word in line.split(' '):
                    S.append(word)

        # Writing the total number of events in the development set |S| to the output file.
        f.write(f'#Output7\t{len(S)}\n')

        ## Lidstone model training

        # Splitting the development set into a training set with exactly the first 90% of the words in S and a
        # validation set with the rest 10% of the words.
        cutoff = round(0.9 * len(S))
        training_set = S[:cutoff]
        val_set = S[cutoff:]

        # Writing the number of events in the validation set to the output file.
        f.write(f'#Output8\t{len(val_set)}\n')
        # Writing the number of events in the training set to the output file.
        f.write(f'#Output9\t{len(training_set)}\n')

        # Using a Counter to get all the unique words (events) in the training set and the number of their occurrences.
        train_unique_events = Counter(training_set)
        # The number of unique words (events) in the training set.
        count_train_unique_events = len(train_unique_events.keys())

        # Writing the number of different events in the training set (i.e. observed vocabulary) to the output file.
        f.write(f'#Output10\t{count_train_unique_events}\n')
        # Writing the number of times the event INPUT WORD appears in the training set to the output file.
        f.write(f'#Output11\t{train_unique_events.get(INPUT_WORD, 0.0)}\n')

        # Writing P(Event = INPUT WORD) the Maximum Likelihood Estimate (MLE) based on the training set
        # (i.e. no smoothing) to the output file.
        f.write(f'#Output12\t{train_unique_events.get(INPUT_WORD, 0.0) / len(training_set)}\n')

        # Writing P(Event = ’unseen-word’) the Maximum Likelihood Estimate (MLE) based on the training set
        # (i.e. no smoothing) to the output file.
        f.write(f'#Output13\t{train_unique_events.get("unseen-word", 0.0) / len(training_set)}\n')

        # Defining new instances of the LidstoneSmoothingModel according to the lambda values.
        lidstone_model_0_01 = LidstoneSmoothingModel(0.01, len(training_set), train_unique_events)
        lidstone_model_0_10 = LidstoneSmoothingModel(0.1, len(training_set), train_unique_events)
        lidstone_model_1_00 = LidstoneSmoothingModel(1.0, len(training_set), train_unique_events)

        # Writing P(Event = INPUT WORD) as estimated by lidstone_model_0_01 model using λ = 0.10 to the output file.
        f.write(f'#Output14\t{lidstone_model_0_10.calc_lidstone_prob(INPUT_WORD)}\n')
        # Writing P(Event = ’unseen-word’) as estimated by lidstone_model_0_01 model using λ = 0.10 to the output file.
        f.write(f'#Output15\t{lidstone_model_0_10.calc_lidstone_prob("unseen-word")}\n')

        # Using a Counter to get all the unique words (events) in the validation set and the number of their occurrences.
        val_unique_events = Counter(val_set)

        # Writing the perplexity on the validation set using λ = 0.01 to the output file.
        f.write(f'#Output16\t{lidstone_model_0_01.perplexity(val_unique_events)}\n')
        # Writing the perplexity on the validation set using λ = 0.10 to the output file.
        f.write(f'#Output17\t{lidstone_model_0_10.perplexity(val_unique_events)}\n')
        # Writing the perplexity on the validation set using λ = 1.00 to the output file.
        f.write(f'#Output18\t{lidstone_model_1_00.perplexity(val_unique_events)}\n')

        # Getting the value of λ that minimizes the perplexity on the validation set and the value of the minimized
        # perplexity using the best value of λ.
        best_lambda_param, best_lambda_perplexity = find_best_lambda_param(len(training_set), train_unique_events,
                                                                           val_unique_events)

        # Writing the value of λ found to minimize the perplexity on the validation set to the output file.
        f.write(f'#Output19\t{best_lambda_param}\n')
        # Writing the minimized perplexity on the validation set using the best value of λ to the output file.
        f.write(f'#Output20\t{best_lambda_perplexity}\n')

        ## Held out model training

        # Splitting the development set into a training set to two halves.
        cutoff = round(0.5 * len(S))
        heldout_training_set = S[:cutoff]
        heldout_val_set = S[cutoff:]

        # Writing the number of events in the validation set to the output file.
        f.write(f'#Output21\t{len(heldout_val_set)}\n')
        # Writing the number of events in the training set to the output file.
        f.write(f'#Output22\t{len(heldout_training_set)}\n')

        # Defining new instances of the HeldoutSmoothingModel according to the lambda values.
        heldout_model = HeldoutSmoothingModel(Counter(heldout_training_set), Counter(heldout_val_set))

        # Writing P(Event = INPUT WORD) as estimated by heldout_model model to the output file.
        f.write(f'#Output23\t{heldout_model.calc_heldout_prob(INPUT_WORD)}\n')
        # Writing P(Event = ’unseen-word’) as estimated by heldout_model model to the output file.
        f.write(f'#Output24\t{heldout_model.calc_heldout_prob("unseen-word")}\n')


        ## Test probabilities sum to 1

        lidstone_model_0_01.test_probabilities_sum_to_1()
        lidstone_model_0_10.test_probabilities_sum_to_1()
        lidstone_model_1_00.test_probabilities_sum_to_1()
        heldout_model.test_probabilities_sum_to_1()
