import math
import sys
from abc import abstractmethod, ABC
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from language_models import LidstoneSmoothingModel
import numpy as np

""""""""""""""""""""""""""""""""""""""
#     Dorin Keshales    313298424
#     Eran Hirsch       302620745
""""""""""""""""""""""""""""""""""""""

EPSILON_THRESHOLD = np.exp(-6)
DEFAULT_K = 10
# TODO: Experiment with several lambda values.
LAMBDA_PARAM = 0.1

# Input arguments
development_set_filename = "dataset/develop.txt"

# TODO: Make sure vocab size is 6800

@dataclass
class Article:
    words_counter: Counter

@dataclass()
class Cluster:
    cluster_id: int
    articles: List[Article]
    cluster_params: 'ClusterParams'

@dataclass()
class ClusterParams:
    lidstone_model: Optional[LidstoneSmoothingModel] = None
    non_normalized_ln_alpha_prior: Optional[float] = None
    normalized_ln_alpha_prior: Optional[float] = None

    def update(self, articles: List[Article], num_of_articles: int):
        self.lidstone_model = self._calc_lidstone_model(articles)
        self.non_normalized_ln_alpha_prior = self._calc_alpha_prior(articles, num_of_articles)

    def _calc_lidstone_model(self, articles: List[Article]) -> LidstoneSmoothingModel:
        # Calculating a counter per cluster, in order to get the training set stats for the lidstone model.
        cluster_words_counter = Counter()
        for article in articles:
            for word, word_count in article.words_counter.items():
                cluster_words_counter[word] += word_count

        # The cluster's lidstone will be used later in equation (5) for the prob of word k in cluster i.
        lidstone_model = LidstoneSmoothingModel(LAMBDA_PARAM, sum(cluster_words_counter.values()),
                                                cluster_words_counter)
        # lidstone_model.test_probabilities_sum_to_1()

        return lidstone_model

    def _calc_alpha_prior(self, articles: List[Article], num_of_articles: int):
        # TODO: Check if alpha should be default ?

        # Calculating alpha_i for the current cluster (formula 2 in the supplemental material).
        alpha_prior = len(articles) / num_of_articles
        if alpha_prior < EPSILON_THRESHOLD:
            alpha_prior = EPSILON_THRESHOLD
        ln_alpha_prior = math.log(alpha_prior)

        return ln_alpha_prior

    def set_normalized_alpha(self, all_alphas_sum: float):
        self.normalized_ln_alpha_prior = self.non_normalized_ln_alpha_prior / all_alphas_sum


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


def _create_clusters(articles: List[Article], num_of_articles: int) -> List[Cluster]:

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
        empty_params = ClusterParams()
        clusters_objs.append(Cluster(cluster_id, articles, empty_params))

    # Activate once the m_step after the initial random selection to initialize the empty params
    _m_step(clusters_objs, num_of_articles)

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
    clusters = _create_clusters(filtered_articles, num_of_articles)

    # Returning the initialised 9 clusters + the total number of articles in the develop.txt file.
    return clusters, num_of_articles


def calc_e_step_z_i(cluster: Cluster, article: Article) -> float:

    """
    Calculation of w_t_i (formula 1 in the supplemental material) in the E step.
    ** Clearer formulas appear in lecture No. 8.
    """

    pi_sum_result = 0.0
    # TODO: Check if ok to run over all words in the document instead of all words in vocab
    for word_k, word_k_count in article.words_counter.items():

        # The frequency of word k in document t.
        n_t_k = word_k_count

        # Calculation of P_ik using Lidstone smoothing (formula 3->5 in the supplemental material).
        prob_of_word_k_in_cluster_i = cluster.cluster_params.lidstone_model.calc_prob(word_k)  # P_ik
        n_t_k_times_ln_prob_of_word_k_in_cluster_i = n_t_k * math.log(prob_of_word_k_in_cluster_i)

        pi_sum_result += n_t_k_times_ln_prob_of_word_k_in_cluster_i

    # The numerator of w_t_i (formula 1 in the supplemental material with underflow handling).
    numerator = cluster.cluster_params.normalized_ln_alpha_prior + pi_sum_result

    # Return the numerator.
    return numerator


def e_step_per_article(article: Article, all_clusters: List[Cluster], k: float=DEFAULT_K) -> List[float]:
    w_ti_per_cluster = []
    clusters_z_i = []
    for cluster in all_clusters:
        z_i = calc_e_step_z_i(cluster, article)
        clusters_z_i.append(z_i)

    max_z_i = max(clusters_z_i)

    clusters_numerators = []
    for cluster_z_i in clusters_z_i:
        if cluster_z_i - max_z_i < -k:
            cluster_numerator = 0
        else:
            cluster_numerator = np.exp(cluster_z_i - max_z_i)
        clusters_numerators.append(cluster_numerator)

    all_clusters_denominator = sum(clusters_numerators)

    for cluster_numerator in clusters_numerators:
        w_ti_per_cluster.append(cluster_numerator / all_clusters_denominator)

    return w_ti_per_cluster


def likelihood():
    pass
    # # Likelihood
    # new_likelihood = 0.0
    # for cluster_idx, numerators in numerators_per_cluster.items():
    #     new_likelihood += np.log(sum(numerators))


def _e_step(clusters: List[Cluster]) -> List[Cluster]:
    # Performing the E step on each one of the clusters.
    new_clusters = defaultdict(list)
    for cluster in clusters:
        for article in cluster.articles:
            # TODO: Check if we should really simply sum the clusters_numerators (z_it=z_i)
            w_ti_per_cluster = e_step_per_article(article, clusters)
            # TODO: Check if the denominator is necessary because it is irrelevant when using argmax
            new_cluster_idx_for_article = np.argmax(w_ti_per_cluster) + 1
            new_clusters[new_cluster_idx_for_article].append(article)

    cluster_by_cluster_id = {cluster.cluster_id: cluster for cluster in clusters}
    new_clusters_objs = []
    # Creating Cluster object for each of the 9 clusters.
    for cluster_id, articles in new_clusters.items():
        cluster = cluster_by_cluster_id[cluster_id]
        old_params = cluster.cluster_params
        new_clusters_objs.append(Cluster(cluster_id, articles, old_params))

    return new_clusters_objs


def _m_step(new_clusters: List[Cluster], num_of_articles: int):
    for cluster in new_clusters:
        cluster.cluster_params.update(cluster.articles, num_of_articles)

    # Normalize alphas
    all_alphas_sum = sum([cluster_obj.cluster_params.non_normalized_ln_alpha_prior for cluster_obj in new_clusters])
    for cluster in new_clusters:
        cluster.cluster_params.set_normalized_alpha(all_alphas_sum)


if __name__ == '__main__':

    clusters, total_num_of_articles = read_file_and_pull_out_events(development_set_filename, '<TRAIN')

    prev_likelihood = None
    while True:
        # TODO: Check probabilities sum to 1 (alpha, P, w_ti)

        new_likelihood = likelihood()
        # TODO: Remove assert
        if prev_likelihood:
            assert new_likelihood > prev_likelihood
        prev_likelihood = new_likelihood

        new_clusters = _e_step(clusters)
        _m_step(new_clusters, total_num_of_articles)
