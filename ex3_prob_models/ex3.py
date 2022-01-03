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

# Parameters of the EM algorithm
EPSILON_THRESHOLD = np.exp(-6)  # TODO: Find best possible epsilon value.
DEFAULT_K = 10
LAMBDA_PARAM = 0.1  # TODO: Experiment with several lambda values.

# Input arguments
development_set_filename = "dataset/develop.txt"


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
    """
    Class the handles the params - alpha and P_i_k.
    """
    lidstone_model: Optional[LidstoneSmoothingModel] = None
    non_normalized_ln_alpha_prior: Optional[float] = None
    normalized_ln_alpha_prior: Optional[float] = None

    def update(self, articles: List[Article], num_of_articles: int):
        """
        Updating the parameters of the EM algorithm - alpha and P_i_k (P_i_k is updated by updating the lidstone model).
        """
        self.lidstone_model = self._calc_lidstone_model(articles)
        self.non_normalized_ln_alpha_prior = self._calc_alpha_prior(articles, num_of_articles)

    def _calc_lidstone_model(self, articles: List[Article]) -> LidstoneSmoothingModel:

        # Calculating a counter per cluster, in order to get the training set stats for the lidstone model.
        cluster_words_counter = Counter()
        for article in articles:
            for word, word_count in article.words_counter.items():
                cluster_words_counter[word] += word_count

        # The cluster's lidstone will be used later in equation (5) for P_i_k (the prob of word k in cluster i).
        lidstone_model = LidstoneSmoothingModel(LAMBDA_PARAM, sum(cluster_words_counter.values()),
                                                cluster_words_counter)

        # lidstone_model.test_probabilities_sum_to_1()
        return lidstone_model

    def _calc_alpha_prior(self, articles: List[Article], num_of_articles: int):
        """
        Calculating alpha_i and handling underflow via the threshold solution suggested in the supplemental material.
        # TODO: Check if alpha should be default? Probably not.
        """
        # TODO: Update this comment?
        # Calculating alpha_i for the current cluster (formula 2 in the supplemental material).
        alpha_prior = len(articles) / num_of_articles

        # If alpha_j becomes less than the threshold we fix alpha_j to be the threshold.
        if alpha_prior < EPSILON_THRESHOLD:
            alpha_prior = EPSILON_THRESHOLD

        # Calculating ln(alpha_i) as requested for calculating z_i.
        ln_alpha_prior = math.log(alpha_prior)

        return ln_alpha_prior

    def set_normalized_alpha(self, all_alphas_sum: float):
        """
        After applying the fix to alpha we need to make sure that Σi alpha_i = 1.
        This is done by re-computing for all j the normalized value of alpha_j.
        """
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
    # TODO: Make sure vocab size after filtering rare words is 6800
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

    # Activate once the m_step after the initial creation of the clusters to initialize the empty params.
    _m_step(clusters_objs, num_of_articles)

    return clusters_objs


def read_file_and_pull_out_events(file_name: str, prefix: str) -> Tuple[List[Cluster], int]:

    # Opening the requested file.
    with open(file_name, 'r', encoding='utf-8') as dev:
        # Ignoring the newline character (\n) at the end of each line.
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
    # Calculating the numerator of w_t_i (formula 1 in the supplemental material) in the E step. # TODO: Delete comment.
    Handling Underflow in the E step by calculating z_i in order to calculate w_t_i.

    ** Clearer formulas appear in lecture No. 8.
    """

    sum_of_n_t_k_times_ln_prob_of_word_k_in_cluster_i = 0.0

    # TODO: Check if it is ok to run over all words in the document instead of all words in vocab.
    for word_k, word_k_count in article.words_counter.items():
        # The frequency of word k in document t.
        n_t_k = word_k_count

        # Calculation of P_i_k using Lidstone smoothing (formula 3->5 in the supplemental material).
        prob_of_word_k_in_cluster_i = cluster.cluster_params.lidstone_model.calc_prob(word_k)  # P_i_k
        n_t_k_times_ln_prob_of_word_k_in_cluster_i = n_t_k * math.log(prob_of_word_k_in_cluster_i)  # n_t_k * ln(P_i_k)

        sum_of_n_t_k_times_ln_prob_of_word_k_in_cluster_i += n_t_k_times_ln_prob_of_word_k_in_cluster_i

    # Calculating z_i.
    z_i = cluster.cluster_params.normalized_ln_alpha_prior + sum_of_n_t_k_times_ln_prob_of_word_k_in_cluster_i

    # Return z_i.
    return z_i


def e_step_per_article(article: Article, all_clusters: List[Cluster], k: float = DEFAULT_K) -> List[float]:
    """
    Calculating w_t_i for each of the clusters as part of the E step of the EM algorithm.

    Note that w_t_i is computed here with respect to the underflow handling instructions listed in the
    supplemental material.
    """
    clusters_z_i = []

    # Calculating z_i for each cluster.
    for cluster in all_clusters:
        z_i = calc_e_step_z_i(cluster, article)
        clusters_z_i.append(z_i)

    # Calculating m = max_i(z_i).
    m = max(clusters_z_i)

    w_t_i_numerators = []

    # Approximating w_t_i according to formula 4 in the supplemental material.
    for cluster_z_i in clusters_z_i:

        # If z_i - m < -k then according to formula 4, the numerator of w_t_i should be 0.
        if cluster_z_i - m < -k:
            w_t_i_numerator = 0
        else:  # Otherwise, the numerator of w_t_i should be e^(z_i) − m.
            w_t_i_numerator = np.exp(cluster_z_i - m)

        w_t_i_numerators.append(w_t_i_numerator)

    # Calculating the numerator of w_t_i (which is the sum of all w_t_i_numerators).
    w_t_i_denominator = sum(w_t_i_numerators)

    w_ti_per_cluster = []
    # Calculating w_t_i per cluster.
    for w_t_i_numerator in w_t_i_numerators:
        w_ti_per_cluster.append(w_t_i_numerator / w_t_i_denominator)

    return w_ti_per_cluster


def likelihood():
    pass

    # # Likelihood
    # new_likelihood = 0.0
    # for cluster_idx, numerators in numerators_per_cluster.items():
    #     new_likelihood += np.log(sum(numerators))


def _e_step(clusters: List[Cluster]) -> List[Cluster]:
    """
    By computing the w_t_i prob for each cluster, we get the new division of documents into clusters.
    """

    # Performing the E step on each one of the clusters.
    new_clusters = defaultdict(list)
    for cluster in clusters:
        for article in cluster.articles:
            # TODO: Check if we should really simply sum the clusters_numerators (z_i_t = z_i)
            w_ti_per_cluster = e_step_per_article(article, clusters)
            # TODO: Check if the denominator is necessary, because it is irrelevant when using argmax.
            new_cluster_idx_for_article = np.argmax(w_ti_per_cluster) + 1
            new_clusters[new_cluster_idx_for_article].append(article)

    cluster_by_cluster_id = {cluster.cluster_id: cluster for cluster in clusters}
    new_clusters_objs = []
    # TODO: Change name to updated_clusters_obj? (we agreed that we don't create new Cluster object each time,
    #  but it the for loop it seems that we still create new Cluster objects).
    # Creating Cluster object for each of the 9 clusters.
    for cluster_id, articles in new_clusters.items():
        cluster = cluster_by_cluster_id[cluster_id]
        old_params = cluster.cluster_params
        new_clusters_objs.append(Cluster(cluster_id, articles, old_params))  # TODO: Here.

    # Returning the updated clusters objects.
    return new_clusters_objs


def _m_step(new_clusters: List[Cluster], num_of_articles: int):
    """
    Performing the M step of the EM algorithm - updating the parameters.
    """

    # Updating th parameters alpha_i and P_i_k.
    for cluster in new_clusters:
        cluster.cluster_params.update(cluster.articles, num_of_articles)

    # Normalizing the alphas.
    all_alphas_sum = sum([cluster_obj.cluster_params.non_normalized_ln_alpha_prior for cluster_obj in new_clusters])
    for cluster in new_clusters:
        cluster.cluster_params.set_normalized_alpha(all_alphas_sum)


if __name__ == '__main__':

    # Initializing 9 clusters.
    clusters, total_num_of_articles = read_file_and_pull_out_events(development_set_filename, '<TRAIN')

    prev_likelihood = None

    while True:

        new_likelihood = likelihood()  # Calculating the likelihhod.
        if prev_likelihood:
            assert new_likelihood > prev_likelihood  # TODO: Remove assert
        prev_likelihood = new_likelihood

        new_clusters = _e_step(clusters)  # Performing the E step of the EM algorithm.
        _m_step(new_clusters, total_num_of_articles)  # Performing the M step of the EM algorithm.

        # TODO: Check probabilities sum to 1 (alpha, P, w_ti)
