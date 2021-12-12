import math
import sys
from collections import Counter
from typing import List

from language_models import LidstoneSmoothingModel, HeldoutSmoothingModel, V

""""""""""""""""""""""""""""""""""""""
#     Dorin Keshales    313298424
#     Eran Hirsch       302620745
""""""""""""""""""""""""""""""""""""""

# Command-line arguments
development_set_filename = sys.argv[1]
test_set_filename = sys.argv[2]
INPUT_WORD = sys.argv[3]
output_filename = sys.argv[4]


def find_best_lambda_param(train_count_events: int, train_unique_events: Counter, val_unique_events: Counter):
    best_lambda = None
    best_lambda_perplexity = math.inf

    # Iterating over a range of optional λ values between 0 and 2
    for lambda_param_int in range(0, 200, 1):

        lambda_param = lambda_param_int / 100
        if lambda_param == 0: continue  # Ignoring λ = 0, since log is not defined for 0 value

        # Calculating the perplexity of a lidstone model
        model = LidstoneSmoothingModel(lambda_param, train_count_events, train_unique_events)
        curr_perplexity = model.perplexity(val_unique_events)

        # Checking if we found a better perplexity than before
        was_better_perplexity_found = curr_perplexity < best_lambda_perplexity

        # If we did, update the best lambda and best perplexity values
        if best_lambda is None or was_better_perplexity_found:
            best_lambda = lambda_param
            best_lambda_perplexity = curr_perplexity

    # Returning the best lambda value and the perplexity value calculated with the best lambda value
    return best_lambda, best_lambda_perplexity


def read_file_and_pull_out_events(file_name: str, prefix: str) -> List[str]:
    # Opening the requested file
    with open(file_name, 'r', encoding='utf-8') as dev:

        # Ignoring the newline character (\n) at the end of each line
        file = [x[:-2] if x.endswith("\n") else x for x in dev.readlines()]

    S = []
    for line in file:
        if line.startswith(prefix) or not line:  # Ignoring header lines.
            continue
        else:
            for word in line.split(' '):
                S.append(word)

    # Returning all relevant events
    return S


if __name__ == '__main__':

    # Opening the output file
    with open(output_filename, 'w', encoding='utf-8') as f:

        """"" Init """""

        # Writing the students info to the output file
        f.write('#Students\tDorin Keshales\tEran Hirsch\t313298424\t302620745\n')
        # Writing the development set file name to the output file
        f.write(f'#Output1\t{development_set_filename}\n')
        # Writing the test set file name to the output file
        f.write(f'#Output2\t{test_set_filename}\n')
        # Writing the INPUT_WORD to the output file
        f.write(f'#Output3\t{INPUT_WORD}\n')
        # Writing the output file name to the output file
        f.write(f'#Output4\t{output_filename}\n')

        # Writing the language vocabulary size to the output file
        f.write(f'#Output5\t{V}\n')

        # Writing the probability of P_uniform(Event = INPUT_WORD) to the output file
        f.write(f'#Output6\t{1 / V}\n')

        """"" Development set preprocessing """""

        # Reading the development set file and pulling out the words only in the relevant lines
        S = read_file_and_pull_out_events(development_set_filename, '<TRAIN')

        # Writing the total number of events in the development set |S| to the output file
        f.write(f'#Output7\t{len(S)}\n')

        """"" Lidstone model training """""

        # Splitting the development set into a training set with exactly the first 90% of the words in S and a
        # validation set with the rest 10% of the words
        cutoff = round(0.9 * len(S))
        training_set = S[:cutoff]
        val_set = S[cutoff:]

        # Writing the number of events in the validation set to the output file
        f.write(f'#Output8\t{len(val_set)}\n')
        # Writing the number of events in the training set to the output file
        f.write(f'#Output9\t{len(training_set)}\n')

        # Using a Counter to get all the unique words (events) in the training set and the number of their occurrences
        train_unique_events = Counter(training_set)
        # The number of unique words (events) in the training set
        count_train_unique_events = len(train_unique_events.keys())

        # Writing the number of different events in the training set (i.e. observed vocabulary) to the output file
        f.write(f'#Output10\t{count_train_unique_events}\n')
        # Writing the number of times the eval_setvent INPUT_WORD appears in the training set to the output file
        f.write(f'#Output11\t{train_unique_events.get(INPUT_WORD, 0.0)}\n')

        # Writing P(Event = INPUT_WORD) the Maximum Likelihood Estimate (MLE) based on the training set
        # (i.e. no smoothing) to the output file
        f.write(f'#Output12\t{train_unique_events.get(INPUT_WORD, 0.0) / len(training_set)}\n')

        # Writing P(Event = ’unseen-word’) the Maximum Likelihood Estimate (MLE) based on the training set
        # (i.e. no smoothing) to the output file
        f.write(f'#Output13\t{train_unique_events.get("unseen-word", 0.0) / len(training_set)}\n')

        # Defining new instances of the LidstoneSmoothingModel according to the lambda values
        lidstone_model_0_01 = LidstoneSmoothingModel(0.01, len(training_set), train_unique_events)
        lidstone_model_0_10 = LidstoneSmoothingModel(0.1, len(training_set), train_unique_events)
        lidstone_model_1_00 = LidstoneSmoothingModel(1.0, len(training_set), train_unique_events)

        # Writing P(Event = INPUT_WORD) as estimated by lidstone_model_0_01 model using λ = 0.10 to the output file
        f.write(f'#Output14\t{lidstone_model_0_10.calc_prob(INPUT_WORD)}\n')
        # Writing P(Event = ’unseen-word’) as estimated by lidstone_model_0_01 model using λ = 0.10 to the output file
        f.write(f'#Output15\t{lidstone_model_0_10.calc_prob("unseen-word")}\n')

        # Using a Counter to get all the unique words (events) in the validation set and the number of their occurrences
        val_unique_events = Counter(val_set)

        # Writing the perplexity on the validation set using λ = 0.01 to the output file
        f.write(f'#Output16\t{lidstone_model_0_01.perplexity(val_unique_events)}\n')
        # Writing the perplexity on the validation set using λ = 0.10 to the output file
        f.write(f'#Output17\t{lidstone_model_0_10.perplexity(val_unique_events)}\n')
        # Writing the perplexity on the validation set using λ = 1.00 to the output file
        f.write(f'#Output18\t{lidstone_model_1_00.perplexity(val_unique_events)}\n')

        # Getting the value of λ that minimizes the perplexity on the validation set and the value of the minimized
        # perplexity using the best value of λ
        best_lambda_param, best_lambda_perplexity = find_best_lambda_param(len(training_set), train_unique_events,
                                                                           val_unique_events)

        # Writing the value of λ found to minimize the perplexity on the validation set to the output file
        f.write(f'#Output19\t{best_lambda_param}\n')
        # Writing the minimized perplexity on the validation set using the best value of λ to the output file
        f.write(f'#Output20\t{best_lambda_perplexity}\n')

        """"" Held out model training """""

        # Splitting the development set into a training set to two halves
        cutoff = round(0.5 * len(S))
        heldout_training_set = S[:cutoff]
        heldout_val_set = S[cutoff:]

        # Writing the number of events in the training set to the output file
        f.write(f'#Output21\t{len(heldout_training_set)}\n')
        # Writing the number of events in the validation (held-out) set to the output file
        f.write(f'#Output22\t{len(heldout_val_set)}\n')

        # Defining new instances of the HeldoutSmoothingModel
        heldout_model = HeldoutSmoothingModel(Counter(heldout_training_set), Counter(heldout_val_set))

        # Writing P(Event = INPUT_WORD) as estimated by the heldout_model model to the output file
        f.write(f'#Output23\t{heldout_model.calc_prob(INPUT_WORD)}\n')
        # Writing P(Event = ’unseen-word’) as estimated by the heldout_model model to the output file
        f.write(f'#Output24\t{heldout_model.calc_prob("unseen-word")}\n')

        # """"" Test probabilities sum to 1 """""
        #
        # lidstone_model_0_01.test_probabilities_sum_to_1()
        # lidstone_model_0_10.test_probabilities_sum_to_1()
        # lidstone_model_1_00.test_probabilities_sum_to_1()
        # heldout_model.test_probabilities_sum_to_1()

        """"" Models evaluation on test set """""

        # Reading the test set file and pulling out the words only in the relevant lines
        test_set = read_file_and_pull_out_events(test_set_filename, '<TEST')

        # Writing the total number of events in the test set to the output file
        f.write(f'#Output25\t{len(test_set)}\n')

        # Defining new instance of the LidstoneSmoothingModel according to the best lambda value found during development
        lidstone_model_best_lambda = LidstoneSmoothingModel(best_lambda_param, len(training_set), train_unique_events)

        # Using a Counter to get all the unique words (events) in the test set and the number of their occurrences
        test_unique_events = Counter(test_set)

        # Calculating the perplexity on the test set according to each one of the models
        lidstone_model_test_perplexity = lidstone_model_best_lambda.perplexity(test_unique_events)
        heldout_model_test_perplexity = heldout_model.perplexity(test_unique_events)

        # Writing the perplexity on the test set, according to the Lidstone model with the best lambda, to the output file
        f.write(f'#Output26\t{lidstone_model_test_perplexity}\n')
        # Writing the perplexity on the test set, according to the held-out model, to the output file
        f.write(f'#Output27\t{heldout_model.perplexity(test_unique_events)}\n')

        # Writing the string ’L’ to the output file if the  Lidstone model is a better language model for the test set
        # than the held-out model; otherwise writing the string ’H’ to the output file
        letter_to_write = "L" if lidstone_model_test_perplexity < heldout_model_test_perplexity else "H"
        f.write(f'#Output28\t{letter_to_write}\n')

        # Writing table comparing different r values
        f.write(f'#Output29\n')
        for r in range(10):
            f_lambda = lidstone_model_best_lambda.calc_f_lambda(r)
            f_H, N_r, t_r = heldout_model.calc_values_for_output_29(r)
            f.write(f'{r}\t{f_lambda}\t{f_H}\t{N_r}\t{t_r}\n')
