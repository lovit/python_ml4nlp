## Document clustering

문서 군집화를 위해서 spherical k-means 를 이용합니다. 그런데 scikit-learn 에는 Euclidean distance 를 이용하는 k-means 만 구현되어 있습니다. [soyclustering][soyclustering] 을 이용하여 spherical k-means 를 학습합니다.

soyclustering 에는 clustering labeling 과 cluster centroid vectors 의 pairwise distance matrix 시각화 방법이 함께 구현되어 있습니다.

`day_7_doc2vec_clustering_(movie_clustering).ipynb` 에서 Doc2Vec 을 이용하여 학습한 영화들을 군집화 합니다.

`day_7_news_clustering.ipynb` 에서 하루의 뉴스에 spherical k-means 를 적용한 뒤, soyclustering 을 이용하여 cluster labeling 을 수행합니다. kmeans_to_pyldavis 를 이용하여 k-means 의 학습결과를 PyLDAVis 로 시각화 합니다.


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

5. [soyclustering][soyclustering]

[soyclustering]: https://github.com/lovit/clustering4docs

```
git clone https://github.com/lovit/clustering4docs
```

6. [kmeans_to_pyldavis][kmeans_to_pyldavis]

[kmeans_to_pyldavis]: https://github.com/lovit/kmeans_to_pyldavis

```
git clone https://github.com/lovit/kmeans_to_pyldavis
```