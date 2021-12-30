import math
import sys
from abc import abstractmethod, ABC
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple
from language_models import LidstoneSmoothingModel

""""""""""""""""""""""""""""""""""""""
#     Dorin Keshales    313298424
#     Eran Hirsch       302620745
""""""""""""""""""""""""""""""""""""""

# Input arguments
development_set_filename = "dataset/develop.txt"


@dataclass
class Article:
    words_counter: Counter


@dataclass()
class Cluster:
    cluster_id: int
    articles: List[Article]


def _parse_input_file_to_articles(file: List[str], prefix: str) -> List[Article]:

    """
    Reading the articles from develop.txt and storing the words of each article in an Article object.
    """

    articles = []

    for line in file:
        if line.startswith(prefix) or not line:  # Ignoring header lines.
            continue
        else:
            article_words = Counter()
            for word in line.split(' '):
                article_words[word] += 1  # Storing the frequency of each word in the article.

            # Make each article as object of type Article.
            article = Article(article_words)
            articles.append(article)

    return articles


def _filter_rare_words(articles: List[Article]) -> List[Article]:

    """
    Filtering rare words (a word that occurs 3 times or less) from the input corpus (develop.txt).
    """

    counter = Counter()

    # Get all the words in the input corpus and their frequency.
    for article in articles:
        for word, word_count in article.words_counter.items():
            counter[word] += word_count

    # Keep only the common words in the input corpus.
    common_words = [word for word, count in counter.items() if count > 3]

    updated_articles = []

    # Update the words in each article following the filtering of the rare words.
    for article in articles:
        filtered_article_words = Counter()
        for word, word_count in article.words_counter.items():
            if word in common_words:
                filtered_article_words[word] = word_count
        updated_articles.append(Article(filtered_article_words))

    return updated_articles


def _create_clusters(articles: List[Article]) -> List[Cluster]:

    """
    Splitting the articles into 9 initial clusters in a modulo-9 manner, according to the ordinal number of the
    article and not its id.
    """

    running_cluster = 0
    clusters = defaultdict(list)

    # Splitting the articles from develop.txt into 9 initial clusters.
    for article in articles:
        cluster_id = (running_cluster % 9) + 1
        clusters[cluster_id].append(article)
        running_cluster += 1

    clusters_objs = []
    # Creating Cluster object for each of the 9 clusters.
    for cluster_id, articles in clusters.items():
        clusters_objs.append(Cluster(cluster_id, articles))

    return clusters_objs


def read_file_and_pull_out_events(file_name: str, prefix: str) -> Tuple[List[Cluster], int]:

    # Opening the requested file
    with open(file_name, 'r', encoding='utf-8') as dev:
        # Ignoring the newline character (\n) at the end of each line
        file = [x[:-2] if x.endswith("\n") else x for x in dev.readlines()]

    # Parsing the input file into articles and the words of each article.
    articles = _parse_input_file_to_articles(file, prefix)

    # Filtering rare words from the input corpus.
    filtered_articles = _filter_rare_words(articles)
    # The total number of articles in the develop.txt file.
    num_of_articles = len(filtered_articles)

    # Splitting the articles into 9 clusters.
    clusters = _create_clusters(filtered_articles)

    # Returning the initialised 9 clusters + the total number of articles in the develop.txt file.
    return clusters, num_of_articles


def e_step(lidstone_model, alpha_i_prior, article: Article) -> float:

    """
    Calculation of w_t_i (formula 1 in the supplemental material) in the E step.
    ** Clearer formulas appear in lecture No. 8.
    """

    pi_multiplication_result = None
    for word_k, word_k_count in article.words_counter.items():

        # The frequency of word k in document t.
        n_t_k = word_k_count

        # TODO: Handle Underflow problems.
        # Calculation of P_ik using Lidstone smoothing (formula 3->5 in the supplemental material).
        prob_of_word_k_in_cluster_i = lidstone_model.calc_prob(word_k)  # P_ik
        prob_of_word_k_in_cluster_i_pow_n_t_k = math.pow(prob_of_word_k_in_cluster_i, n_t_k)  # (P_ik)^n_t_k

        if pi_multiplication_result is None:
            pi_multiplication_result = prob_of_word_k_in_cluster_i_pow_n_t_k
        else:
            pi_multiplication_result *= prob_of_word_k_in_cluster_i_pow_n_t_k

    # The numerator of w_t_i (formula 1 in the supplemental material).
    numerator = alpha_i_prior * pi_multiplication_result
    # The denominator of w_t_i (formula 1 in the supplemental material).
    denominator =

    # Return the value of w_t_i.
    return numerator / denominator


def e_step_per_cluster(cluster: Cluster, total_num_of_articles: int):

    # TODO: Experiment with several lambda values.
    lambda_param = 0.1

    # Calculating a counter per cluster, in order to get the training set stats for the lidstone model.
    cluster_words_counter = Counter()
    for article in cluster.articles:
        for word, word_count in article.words_counter.items():
            cluster_words_counter[word] += word_count

    # The cluster's lidstone will be used later in equation (5) for the prob of word k in cluster i.
    lidstone_model = LidstoneSmoothingModel(lambda_param, sum(cluster_words_counter.values()), cluster_words_counter)
    # lidstone_model.test_probabilities_sum_to_1()

    # TODO: Handle Underflow problem.
    # Calculating alpha_i for the current cluster (formula 2 in the supplemental material).
    alpha_i_prior = len(cluster.articles) / total_num_of_articles

    # Calculating the E step for each article in the current cluster.
    for article in cluster.articles:
        e_step(lidstone_model, alpha_i_prior, article)


if __name__ == '__main__':

    clusters, total_num_of_articles = read_file_and_pull_out_events(development_set_filename, '<TRAIN')

    # Performing the E step on each one of the clusters.
    for cluster in clusters:
        e_step_per_cluster(cluster, total_num_of_articles)
    print(clusters)
