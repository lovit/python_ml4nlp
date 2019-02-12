from sklearn.metrics import pairwise_distances

def most_similar_terms(term, wv, idx_to_vocab, vocab_to_idx, topn=10):
    """
    Arguments
    ---------
    term : str
        Query term
    wv : numpy.ndarray or scipy.sparse.matrix
        Word representation matrix. shape = (n_terms, dim)
    idx_to_vocab : list of str
        Index to word
    vocab_to_idx : {str:int}
        Word to index
    topn : int
        Number of most similar words
    """

    # encode term as index
    idx = vocab_to_idx.get(term, -1)
    if idx < 0:
        return []
    
    # prepare query term vector
    query_vec = wv[idx,:].reshape(1,-1)

    # compute cosine - distance
    dist = pairwise_distances(
        wv,
        query_vec,
        metric='cosine'
    ).reshape(-1)

    # find most closest terms
    # ignore query term itself
    similar_idx = dist.argsort()[1:topn+1]

    # get their distance
    similar_dist = dist[similar_idx]

    # format : [(term, distance), ... ]
    similar_terms = [(idx, d) for idx, d in zip(similar_idx, similar_dist)]

    # decode term index to vocabulary
    similar_terms = [(idx_to_vocab[idx], d) for idx, d in similar_terms]

    # return
    return similar_terms

def most_similar_docs_from_term(term, wv, dv, vocab_to_idx, topn=10):
    """
    Arguments
    ---------
    term : str
        Query term
    wv : numpy.ndarray or scipy.sparse.matrix
        Word representation matrix. shape = (n_terms, dim)
    dv : numpy.ndarray or scipy.sparse.matrix
        Document representation matrix. shape = (n_docs, dim)
    vocab_to_idx : {str:int}
        Word to index
    topn : int
        Number of most similar documents
    """

    # encode term as index
    idx = vocab_to_idx.get(term, -1)
    if idx < 0:
        return []

    # prepare query term vector
    query_vec = wv[idx,:].reshape(1,-1)

    # compute distance between query term vector and document vectors
    dist = pairwise_distances(
        dv,
        query_vec,
        metric='cosine'
    ).reshape(-1)

    # find similar document indices
    similar_doc_idx = dist.argsort()[:topn]

    # return
    return similar_doc_idx

def get_bow(doc_idx, bow, idx_to_vocab, topn=10):
    """
    Arguments
    ---------
    term : str
        Query term
    bow : scipy.sparse.matrix
        Term frequency matrix. shape = (n_docs, n_terms)
    idx_to_vocab : list of str
        Index to word
    topn : int
        Number of most frequent terms
    """

    # get term frequency submatrix
    x_sub = bow[doc_idx,:]

    # get term indices and their frequencies
    terms = x_sub.nonzero()[1]
    freqs = x_sub.data

    # format : [(term, frequency), ... ]
    bow = [(term, freq) for term, freq in zip(terms, freqs)]
    
    # sort by frequency in decreasing order
    bow = sorted(bow, key=lambda x:-x[1])[:topn]

    # decode term index to vocabulary
    bow = [(idx_to_vocab[term], freq) for term, freq in bow]

    # return
    return bow