import numpy as np
import pandas as pd
import pyLDAvis
from collections import Counter
from collections import namedtuple
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .utils import _df_topic_coordinate
from .utils import _df_topic_info
from .utils import _df_token_table

def _kmeans_to_prepared_data_proportion_score(bow, index2word,
    centers, labels, embedding_method='tsne', radius=3.5,
    n_candidate_words=50, n_printed_words=30, lambda_step=0.01):

    if embedding_method == 'pca':
        plot_opts={'xlab': 'PCA1', 'ylab': 'PCA2'}
    else:
        plot_opts={'xlab': 't-SNE1', 'ylab': 't-SNE2'}

    n_clusters = centers.shape[0]
    n_docs, n_terms = bow.shape

    cluster_size = Counter(labels)
    cluster_size = np.asarray([cluster_size.get(c, 0) for c in range(n_clusters)])

    term_frequency = np.asarray(bow.sum(axis=0)).reshape(-1)
    term_frequency[np.where(term_frequency == 0)[0]] = 0.01

    weighted_centers = np.zeros((n_clusters, n_terms))
    for c, n_docs in enumerate(cluster_size):
        weighted_centers[c] = centers[c] * n_docs

    # prepare parameters
    topic_coordinates = _get_topic_coordinates(
        centers, cluster_size, radius, embedding_method)

    topic_info = _get_topic_info(
        centers, cluster_size, index2word,
        weighted_centers, term_frequency, n_candidate_words)

    token_table = _get_token_table(
        weighted_centers, topic_info, index2word)

    topic_order = cluster_size.argsort()[::-1].tolist()

    # convert to pandas.DataFrame
    topic_coordinate_df = _df_topic_coordinate(topic_coordinates)
    topic_info_df = _df_topic_info(topic_info)
    token_table_df = _df_token_table(token_table)

    # ready pyLDAvis.PreparedData
    prepared_data = pyLDAvis.PreparedData(
        topic_coordinate_df,
        topic_info_df,
        token_table_df,
        n_printed_words,
        lambda_step,
        plot_opts,
        topic_order
    )

    # return
    return prepared_data

def _get_token_table(weighted_centers, topic_info, index2word):
    TokenTable = namedtuple('TokenTable', 'term Topic Freq Term'.split())

    term_proportion = weighted_centers / weighted_centers.sum(axis=0)

    token_table = []
    for info in topic_info:
        try:
            c = int(info.Category[5:])
        except:
            # Category: Default
            continue
        token_table.append(
            TokenTable(
                info.term,
                c,
                term_proportion[c-1,info.term],
                info.Term
            )
        )

    return token_table

def _get_topic_info(centers, cluster_size, index2word,
    weighted_centers, term_frequency, n_candidate_words=100):

    TopicInfo = namedtuple(
        'TopicInfo',
        'term Category Freq Term Total loglift logprob'.split()
    )

    l1_normalize = lambda x:x/x.sum()
    n_clusters, n_terms = centers.shape

    weighted_center_sum = weighted_centers.sum(axis=0)
    total_sum = weighted_center_sum.sum()
    term_proportion = weighted_centers / weighted_center_sum

    topic_info = []

    # Category: Default
    default_terms = term_frequency.argsort()[::-1][:n_candidate_words]
    default_term_frequency = term_frequency[default_terms]
    default_term_loglift = 15 * default_term_frequency / default_term_frequency.max() + 10
    for term, freq, loglift in zip(default_terms, default_term_frequency, default_term_loglift):
        topic_info.append(
            TopicInfo(
                term,
                'Default',
                0.99,
                index2word[term],
                term_frequency[term],
                loglift,
                loglift
            )
        )

    # Category: for each topic
    for c, n_docs in enumerate(cluster_size):
        if n_docs == 0:
            keywords.append([])
            continue

        topic_idx = c + 1

        n_prop = l1_normalize(weighted_center_sum - (centers[c] * n_docs))
        p_prop = l1_normalize(centers[c])

        indices = np.where(p_prop > 0)[0]
        indices = sorted(indices, key=lambda idx:-p_prop[idx])[:n_candidate_words]
        scores = [(idx, p_prop[idx] / (p_prop[idx] + n_prop[idx])) for idx in indices]

        for term, loglift in scores:
            topic_info.append(
                TopicInfo(
                    term,
                    'Topic%d' % topic_idx,
                    term_proportion[c, term] * term_frequency[term],
                    index2word[term],
                    term_frequency[term],
                    loglift,
                    p_prop[term]
                )
            )

    return topic_info

def _get_topic_coordinates(centers, cluster_size,
    radius=5, embedding_method='tsne'):

    TopicCoordinates = namedtuple(
        'TopicCoordinates',
        'topic x y topics cluster Freq'.split()
    )

    n_clusters = centers.shape[0]

    if embedding_method == 'pca':
        coordinates = _coordinates_pca(centers)
    else:
        coordinates = _coordinates_tsne(centers)

    # scaling
    coordinates = 5 * coordinates / max(coordinates.max(), abs(coordinates.min()))

    cluster_size = np.asarray(
        [np.sqrt(cluster_size[c] + 1) for c in range(n_clusters)])
    cs_min, cs_max = cluster_size.min(), cluster_size.max()
    cluster_size = radius * (cluster_size - cs_min) / (cs_max - cs_min) + 0.2

    topic_coordinates = [
        TopicCoordinates(c+1, coordinates[i,0], coordinates[i,1], i+1, 1, cluster_size[c])
        for i, c in enumerate(sorted(range(n_clusters), key=lambda x:-cluster_size[x]))
    ]

    topic_coordinates = sorted(topic_coordinates, key=lambda x:-x.Freq)
    return topic_coordinates

def _coordinates_pca(centers):
    return PCA(n_components=2).fit_transform(centers)

def _coordinates_tsne(centers):
    return TSNE(n_components=2, metric='cosine').fit_transform(centers)