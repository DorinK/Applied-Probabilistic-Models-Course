import math
import sys
from abc import abstractmethod, ABC
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
from matplotlib import pyplot

from language_models import LidstoneSmoothingModel
import numpy as np
import pandas as pd

""""""""""""""""""""""""""""""""""""""
#     Dorin Keshales    313298424
#     Eran Hirsch       302620745
""""""""""""""""""""""""""""""""""""""

# Parameters of the EM algorithm
EPSILON_THRESHOLD = np.exp(-10)
DEFAULT_K = 10
LAMBDA_PARAM = 0.05
# TODO: Revert threshold
STOPPING_THRESHOLD = 0.05
# STOPPING_THRESHOLD = 10000.0

# Input arguments
development_set_filename = "dataset/develop.txt"


@dataclass
class Article:
    words_counter: Counter


@dataclass()
class Cluster:
    cluster_id: Optional[int] = None
    articles: Optional[List[Article]] = None
    cluster_params: Optional['ClusterParams'] = None

    def update(self, articles: List[Article]):
        self.articles = articles


@dataclass()
class ClusterParams:
    """
    Class the handles the EM params - alpha and P_i_k.
    """
    lidstone_model: Optional[LidstoneSmoothingModel] = None
    non_normalized_ln_alpha_prior: Optional[float] = None
    normalized_ln_alpha_prior: Optional[float] = None

    def update(self, articles: List[Article], num_of_articles: int, vocab_size: int):
        """
        Updating the parameters of the EM algorithm - alpha and P_i_k (P_i_k is updated by updating
        the lidstone model).
        """
        self.lidstone_model = self._calc_lidstone_model(articles, vocab_size)
        self.non_normalized_ln_alpha_prior = self._calc_alpha_prior(articles, num_of_articles)

    def _calc_lidstone_model(self, articles: List[Article], vocab_size: int) -> LidstoneSmoothingModel:
        """
        Calculating the Lidstone model based on the training set and lambda value.
        """
        # Calculating a counter per cluster, in order to get the training set stats for the lidstone model.
        cluster_words_counter = Counter()
        for article in articles:
            for word, word_count in article.words_counter.items():
                cluster_words_counter[word] += word_count

        # The cluster's lidstone will be used later in equation (5) for P_i_k (the prob of word k in cluster i).
        lidstone_model = LidstoneSmoothingModel(LAMBDA_PARAM, sum(cluster_words_counter.values()),
                                                cluster_words_counter, vocab_size)

        # lidstone_model.test_probabilities_sum_to_1()
        return lidstone_model

    def _calc_alpha_prior(self, articles: List[Article], num_of_articles: int):
        """
        Calculating alpha_i and handling underflow via the threshold solution suggested in the supplemental material.
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


def test_alpha_probabilities_sum_to_1(clusters: List[Cluster]):
    all_alphas_sum_after_normalization = sum([cluster.cluster_params.normalized_ln_alpha_prior for cluster in clusters])
    print(f"All alphas sum after normalization: {all_alphas_sum_after_normalization}")


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


def _filter_rare_words(articles: List[Article]) -> Tuple[List[Article], int, int]:
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

    # Calculating the updated vocab size - 6800.
    vocab_size = len(common_words)
    # Calculating the amount of words left in the dataset.
    count_of_words = sum([count for word, count in counter.items() if count > 3])

    return updated_articles, vocab_size, count_of_words


def _create_clusters(articles: List[Article], num_of_articles: int, vocab_size: int) -> List[Cluster]:
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
    _m_step(clusters_objs, num_of_articles, vocab_size)

    return clusters_objs


def preprocessing_the_input_file(file_name: str, prefix: str) -> Tuple[List[Cluster], int, int, int]:
    """
    Reading the input file, dividing them into articles, filtering rare words from the corpus and
    creating 9 initial clusters.
    """

    # Opening the requested file.
    with open(file_name, 'r', encoding='utf-8') as dev:
        # Ignoring the newline character (\n) at the end of each line.
        file = [x[:-2] if x.endswith("\n") else x for x in dev.readlines()]

    # Parsing the input file into articles and the words of each article.
    articles = _parse_input_file_to_articles(file, prefix)

    # Filtering rare words from the input corpus.
    filtered_articles, vocab_size, count_of_words = _filter_rare_words(articles)
    # The total number of articles in the develop.txt file.
    num_of_articles = len(filtered_articles)

    # Splitting the articles into 9 clusters.
    clusters = _create_clusters(filtered_articles, num_of_articles, vocab_size)

    # Returning the initialised 9 clusters + the total number of articles in the develop.txt file.
    return clusters, num_of_articles, vocab_size, count_of_words


def calc_e_step_z_i(cluster: Cluster, article: Article) -> float:
    """
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


def calc_w_t_i_numerators_and_denominator(article: Article, all_clusters: List[Cluster], k: float = DEFAULT_K):
    """
    Calculating the numerators of w_t_i (per cluster) and the denominator of w_t_i.
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

    return w_t_i_numerators, w_t_i_denominator, m


def calc_likelihood(clusters: List[Cluster]) -> float:
    """
    Calculating the log likelihood.
    """
    sum = 0.0

    for cluster in clusters:
        for article in cluster.articles:
            _, w_t_i_denominator, m = calc_w_t_i_numerators_and_denominator(article, clusters)
            sum += m + math.log(w_t_i_denominator)

    # Returning the log likelihood value.
    return sum


def calc_perplexity(likelihood: float, count_of_words: int) -> float:
    """
    Calculating the mean perplexity per word.
    """
    # TODO: Check if this value makes sense.
    return math.exp((-1 / count_of_words) * likelihood)


def e_step_per_article(article: Article, all_clusters: List[Cluster]) -> List[float]:
    """
    Calculating w_t_i for each of the clusters as part of the E step of the EM algorithm.

    Note that w_t_i is computed here with respect to the underflow handling instructions listed in the
    supplemental material.
    """

    # Calculating the numerators and denominator of w_t_i.
    w_t_i_numerators, w_t_i_denominator, _ = calc_w_t_i_numerators_and_denominator(article, all_clusters)

    w_ti_per_cluster = []
    # Calculating w_t_i per cluster.
    for w_t_i_numerator in w_t_i_numerators:
        w_ti_per_cluster.append(w_t_i_numerator / w_t_i_denominator)  # denominator is not necessary here (argmax)

    # print(f"sum(w_ti_per_cluster) {sum(w_ti_per_cluster)}")

    return w_ti_per_cluster


def _e_step(clusters: List[Cluster]):
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

    # Creating Cluster object for each of the 9 clusters.
    for cluster_id, articles in new_clusters.items():
        cluster = cluster_by_cluster_id[cluster_id]
        cluster.update(articles)


def _m_step(new_clusters: List[Cluster], num_of_articles: int, vocab_size: int):
    """
    Performing the M step of the EM algorithm - updating the parameters.
    """

    # Updating the parameters alpha_i and P_i_k.
    for cluster in new_clusters:
        cluster.cluster_params.update(cluster.articles, num_of_articles, vocab_size)

    # Normalizing the alphas.
    all_alphas_sum = sum([cluster_obj.cluster_params.non_normalized_ln_alpha_prior for cluster_obj in new_clusters])
    for cluster in new_clusters:
        cluster.cluster_params.set_normalized_alpha(all_alphas_sum)

    # Checking the probability of alphas sums to 1.
    # test_alpha_probabilities_sum_to_1(new_clusters)


def plot_graph(title, x_label, y_label, indexes, values, plot_color, x_ticks=None):
    """
    Function for plotting graphs.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_label: pyplot.xticks(x_ticks)

    pyplot.plot(indexes, values, color=plot_color)

    plt.savefig("plots/" + title + ".png", dpi=192)


if __name__ == '__main__':

    # Initializing 9 clusters.
    clusters, total_num_of_articles, vocab_size, count_of_words = preprocessing_the_input_file(
        development_set_filename, '<TRAIN')

    num_epochs = 0
    prev_likelihood = None
    prev_perplexity = None
    likelihood_over_epochs = []
    perplexity_over_epochs = []

    while True:

        num_epochs += 1

        new_likelihood = calc_likelihood(clusters)  # Calculating the new likelihood.
        new_perplexity = calc_perplexity(new_likelihood, count_of_words)  # Calculating the new perplexity.

        # Save new likelihood and perplexity values.
        likelihood_over_epochs.append(new_likelihood)
        perplexity_over_epochs.append(new_perplexity)
        print(f"new_likelihood: {new_likelihood}")

        if prev_likelihood:
            assert new_likelihood >= prev_likelihood  # TODO: Remove assert
            assert new_perplexity <= prev_perplexity  # TODO: Remove assert
            # Stop the EM algorithm if the likelihood value converges.
            if new_likelihood - prev_likelihood <= STOPPING_THRESHOLD:
                break

        prev_likelihood = new_likelihood
        prev_perplexity = new_perplexity
        # TODO: Why likelihood is negative?

        _e_step(clusters)  # Performing the E step of the EM algorithm.
        _m_step(clusters, total_num_of_articles, vocab_size)  # Performing the M step of the EM algorithm.

    # Plotting the graphs of the Log Likelihood and Mean Perplexity per Word over epochs.
    plot_graph(title='Log Likelihood over Epochs', x_label="Epoch", y_label="Log Likelihood * 1e-6",
               indexes=[i for i in range(0, num_epochs)], values=[val * 1e-6 for val in likelihood_over_epochs],
               plot_color='steelblue')
    plot_graph(title='Mean Perplexity per Word over Epochs', x_label="Epoch", y_label="Mean Perplexity per Word",
               indexes=[i for i in range(0, num_epochs)], values=perplexity_over_epochs,
               plot_color='indianred')

    print("Done")
