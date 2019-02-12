## Day 6 tutorials

Day 6 에서는 scikit-learn 의 Singular Value Decomposition (SVD), Nonnegative Matrix Factorization (NMF) 와 Gensim 의 Latent Dirichlet Allocation (LDA) 를 이용한 topic modeling 을 실습합니다. 또한 PyLDAVis 를 이용하여 학습된 LDA 를 시각화합니다.

`day_6_0_LSI_using_SVD.ipynb` 에서는 TruncatedSVD 를 이용하여 Latent Semantic Indexing (LSI) 를 학습합니다. Topically similar terms 를 검색하고, 단어와 topically similar 한 문서를 검색합니다.

`day_6_1_topic_modeling_with_NMF.ipynb` 에서는 Nonnegative Matrix Factorization 를 이용하는 topic modeling 을 학습합니다. 학습된 topics 의 중요 단어를 추출하여 topics 의 의미를 해석하고, 한 단어와 관련있는 topics 을 탐색합니다.

`day_6_2_topic_modeling_with_lda.ipynb` 에서는 Gensim 을 이용하여 LDA 를 학습합니다. 학습된 모델에서 topic term probability 를 가져오는 방법과 probability 해석 시 주의해야 할 점에 대하여 알아봅니다.

`day_6_3_pyLDAvis.ipynb` 에서는 `day_6_2` 에서 학습한 LDA model 을 PyLDAVis 를 이용하여 시각화 합니다.

`day_6_a_gensim_dictionary_format.ipynb` 에서는 Gensim 의 Dictionary 형식에 대하여 알아봅니다. LDA 를 학습할 때에는 필요하지 않지만, LDAVis 를 학습할 때에는 추가로 필요합니다.

`day_6_a_gibbs_sampling_and_lda.ipynb` 은 `밑바닥부터 시작하는 데이터과학`의 LDA tutorials 코드입니다. 코드 설명은 저자의 책을 참고하시기 바랍니다.

## Requirements

이 실습 코드에서는 아래의 외부 패키지를 이용합니다.

1. [lovit_textmining_dataset](https://github.com/lovit/textmining_dataset)

```
git clone https://github.com/textmining_dataset
```

2. [Scikit-learn](https://scikit-learn.org/)

```
pip install scikit-learn
```

3. [Gensim](https://radimrehurek.com/gensim/)

```
pip install gensim
```

4. [PyLDAVis](https://github.com/bmabey/pyLDAvis)

```
pip install pyldavis
```