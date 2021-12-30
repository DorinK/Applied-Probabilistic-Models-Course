""""""""""""""""""""""""""""""""""""""
#     Dorin Keshales    313298424
#     Eran Hirsch       302620745

""""""""""""""""""""""""""""""""""""""

import math
import sys
from abc import abstractmethod, ABC
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple
from language_models import LidstoneSmoothingModel


# Input arguments
development_set_filename = "dataset/develop.txt"


@dataclass
class Article:
    words_counter: Counter

@dataclass()
class Cluster:
    cluster_id: int
    articles: List[Article]


def _parse_file_to_articles(file: List[str], prefix: str) -> List[Article]:
    articles = []
    for line in file:
        if line.startswith(prefix) or not line:  # Ignoring header lines.
            continue
        else:
            article_words = Counter()
            for word in line.split(' '):
                article_words[word] += 1

            article = Article(article_words)
            articles.append(article)

    return articles


def _filter_rare_words(articles: List[Article]) -> List[Article]:
    counter = Counter()
    for article in articles:
        for word, word_count in article.words_counter.items():
            counter[word] += word_count

    common_words = [word for word, count in counter.items() if count > 3]

    new_articles = []
    for article in articles:
        filtered_article_words = Counter()
        for word, word_count in article.words_counter.items():
            if word in common_words:
                filtered_article_words[word] = word_count
        new_articles.append(Article(filtered_article_words))

    return new_articles


def _create_clusters(articles: List[Article]) -> List[Cluster]:
    running_cluster = 0
    clusters = defaultdict(list)
    for article in articles:
        cluster_id = (running_cluster % 9) + 1
        clusters[cluster_id].append(article)

        running_cluster += 1

    clusters_objs = []
    for cluster_id, articles in clusters.items():
        clusters_objs.append(Cluster(cluster_id, articles))

    return clusters_objs


def read_file_and_pull_out_events(file_name: str, prefix: str) -> Tuple[List[Cluster], int]:

    # Opening the requested file
    with open(file_name, 'r', encoding='utf-8') as dev:

        # Ignoring the newline character (\n) at the end of each line
        file = [x[:-2] if x.endswith("\n") else x for x in dev.readlines()]

    articles = _parse_file_to_articles(file, prefix)
    articles = _filter_rare_words(articles)
    num_of_articles = len(articles)
    clusters = _create_clusters(articles)

    # Returning all clusters
    return clusters, num_of_articles


# def calc_equation_5():


def e_step(lidstone_model, alpha_i_prior, article: Article) -> float:
    """
    This is function (1) in the supplemental material
    """

    multiplication_result = None
    for word_k, word_k_count in article.words_counter.items():
        n_t_k = word_k_count
        # TODO: Underflow stuff
        # This is function (5) in the supplemental material
        prob_of_word_k_in_cluster_i = lidstone_model.calc_prob(word_k)
        prob_of_word_k_in_cluster_i_pow_n_t_k = math.pow(prob_of_word_k_in_cluster_i, n_t_k)
        if multiplication_result is None:
            multiplication_result = prob_of_word_k_in_cluster_i_pow_n_t_k
        else:
            multiplication_result *= prob_of_word_k_in_cluster_i_pow_n_t_k

    numerator = alpha_i_prior * multiplication_result
    denominator =

    return numerator / denominator


def e_step_per_cluster(cluster: Cluster, total_num_of_articles: int):
    # TODO: Fix lambda
    lambda_param = 0.1

    # Calculate the counter per cluster in order to get the training set stats for the lidstone model
    cluster_words_counter = Counter()
    for article in cluster.articles:
        for word, word_count in article.words_counter.items():
            cluster_words_counter[word] += word_count

    # The cluster's lidstone will be used later in equation (5) for the prob of word k in cluster i
    lidstone_model = LidstoneSmoothingModel(lambda_param, sum(cluster_words_counter.values()), cluster_words_counter)
    # lidstone_model.test_probabilities_sum_to_1()

    # TODO: Underflow stuff
    # This is function (2) in the supplemental material
    alpha_i_prior = len(cluster.articles) / total_num_of_articles

    # Calc e step for each article
    for article in cluster.articles:
        e_step(lidstone_model, alpha_i_prior, article)


if __name__ == '__main__':
    clusters, total_num_of_articles = read_file_and_pull_out_events(development_set_filename, '<TRAIN')

    # Calc cluster for each article
    for cluster in clusters:
        e_step_per_cluster(cluster, total_num_of_articles)
    print(clusters)
