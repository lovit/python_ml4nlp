## Day 5 tutorials

Day 5 에서는 Gensim 을 이용하여 Word2Vec, FastText 와 같은 word embedding 과 Doc2Vec 같은 document embedding 을 연습합니다.

또한 고차원의 벡터를 2 차원으로 압축함으로써 고차원의 벡터를 시각화하는 방법들을 알아봅니다. 또한 plotting 툴로 matplotlib 대신 Bokeh 를 이용합니다. matplotlib 은 제안된지 오래되기도 하였으며 고정된 그림밖에 그릴 수 없습니다. 하지만 최근에 구현된 Plotly 나 Bokeh 는 Java Script 를 이용하여 동적인 plots 그려줍니다.

### Word2Vec, Doc2Vec, FastText

`day_5_0_word2vec_and_doc2vec_moviereview_(gensim_3.6)` 에서는 Gensim 을 이용하여 Word2Vec 과 Doc2Vec 을 학습합니다. 응용에 필요한 parameters 의 위치를 확인하는 연습까지 합니다.

`day_5_4_fasttext_gensim` 에서는 Gensim 의 FastText 를 한국어에 적용합니다. 이를 위해서는 한글의 한 글자를 초/중/종성으로 분리/조합해야 합니다. 이에 대한 연습을 합니다.

`day_5_a_word2vec_news_embedding_(space_tokenization)` 에서는 띄어쓰기 기준으로 토크나이징을 하여 Word2Vec 을 학습합니다. 토크나이징을 하지 않으면 어절 간의 유사어가 학습됩니다. 토크나이저의 필요성에 대하여 생각할 수 있는 예시입니다. 이를 확인해 봅니다.

`day_5_a_fasttext_facebook.ipynb` 에서는 Facebook Research 에서 배포한 fasttext 패키지를 이용하여 unsupervised, supervised FastText 를 모두 학습합니다.

### Explicit word representation

Levy and Goldberg (2014) 는 Word2Vec 의 negative sampling 을 이용하는 Skip-gram 은 word - context cooccurrence matrix 에 PMI 를 적용한 것과 같음을 확인하였습니다. `day_5_5_explicit_word_representation_(pmi_svd)` 은 Levy and Goldberg (2014) 의 내용을 구현합니다.

이를 위해서는 co-occurrence matrix 를 만들 수 있어야 합니다. 이의 내용은 `day_5_a_cooccurrence_matrix` 에서 연습할 수 있습니다.

- Levy, O., & Goldberg, Y. (2014). Neural word embedding as implicit matrix factorization. In Advances in neural information processing systems (pp. 2177-2185).

### Vector visualization (Plotting)

`day_5_1_bokeh_plotting` 에서는 Bokeh 를 `day_5_1_matplotlib_plotting` 에서는 matplotlib 을 이용하여 scatter plot 을 그려봅니다.

고차원 벡터 공간을 시각화 하기 위한 dimension reduction algorithms 는 많습니다. 각 방법들은 각자가 가정하는 공간이 있습니다. 그리고 scikit-learn 에서는 다양한 방법들을 제공합니다. 각 단어에 대한 document frequency vector 를 term representation 으로 이용한 뒤, 이를 2 차원으로 압축하여 고차원의 단어 공간을 시각화 합니다. 이를 위하여 아래 알고리즘들을 살펴봅니다.

- Multi Dimensional Scaling (MDS)
- Printipal Component Analysis (PCA)
- kernel Principal Component Analysis (kPCA)
- t-Stochastic Neighbor Embedding (t-SNE)
- ISOMAP
- Locally Linear Embedding (LLE)

`day_5_2_embedding_for_visualization_bokeh` 에서는 위 방법들을 Bokeh 를 이용하여 시각화하며, `day_5_2_embedding_for_visualization_matplotlib` 에서는 그 중 몇 가지에 대해 matplotlib 을 이용하여 plots 을 그려봅니다.

`day_5_3_visualize_similar_movies_(doc2vec_tsne)` 에서는 Doc2Vec 을 이용하여 학습한 document vectors 를 t-SNE 로 시각화 하여 유사 영화를 시각적으로 확인합니다.


## Requirements

이 실습 코드에서는 아래의 외부 패키지를 이용합니다.

1. [lovit_textmining_dataset](https://github.com/lovit/textmining_dataset)

```
git clone https://github.com/lovit/textmining_dataset
```

2. [soynlp](https://github.com/lovit/soynlp)

학습데이터를 이용하는 한국어 용언 분석기 (형태소 분석기) 입니다. Hidden Markov Model (HMM) 기반 한국어 형태소 분석기의 구현 예제 (4-1) 에서 이용합니다.

```
pip install soynlp
git clone https://github.com/lovit/soynlp
```

3. [Gensim >= 3.6.0](https://radimrehurek.com/gensim/)

```
pip install gensim
```

4. [Bokeh >= 0.13.0](https://bokeh.pydata.org/en/latest/)

```
pip install bokeh
```

5. [Scikit-learn](https://scikit-learn.org/)

```
pip install scikit-learn
```
