from sklearn.preprocessing import normalize
import numpy as np
import pyLDAvis

def _kmeans_to_prepared_data_pyldavis_score(x, index2word,
    centers, labels, embedding_method='tsne', radius=3.5,
    n_candidate_words=50, n_printed_words=30, lambda_step=0.01):
    """
    Dont use pyLDAvis embedding method. It shows unstable training results.
    """

    topic_term_dists = normalize(centers, norm='l1')

    empty_clusters = np.where(topic_term_dists.sum(axis=1) == 0)[0]
    default_weight = 1 / centers.shape[1]
    topic_term_dists[empty_clusters,:] = default_weight

    doc_topic_dists = np.zeros((x.shape[0], centers.shape[0]))
    for d, label in enumerate(labels):
        doc_topic_dists[d,label] = 1

    doc_lengths = x.sum(axis=1).A.ravel()

    term_frequency = x.sum(axis=0).A.ravel()
    term_frequency[term_frequency == 0] = 0.01 # preventing zeros

    if embedding_method == 'tsne':
        return pyLDAvis.prepare(
            topic_term_dists, doc_topic_dists, doc_lengths, index2word, term_frequency,
            R=radius, lambda_step=lambda_step, sort_topics=True,
            plot_opts={'xlab': 't-SNE1', 'ylab': 't-SNE2'}
        )
    else:
        return pyLDAvis.prepare(
            topic_term_dists, doc_topic_dists, doc_lengths, index2word, term_frequency,
            R=radius, lambda_step=lambda_step
        )