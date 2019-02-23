from .proportion import _kmeans_to_prepared_data_proportion_score
from .pyldavis import _kmeans_to_prepared_data_pyldavis_score


def kmeans_to_prepared_data(bow, index2word, centers, labels,
    embedding_method='tsne', radius=3.5, n_candidate_words=50,
    n_printed_words=30, lambda_step=0.01):

    return _kmeans_to_prepared_data_proportion_score(
        bow, index2word, centers, labels, embedding_method,
        radius, n_candidate_words, n_printed_words, lambda_step)
