## k-means to pyLDAvis

pyLDAvis 를 이용하여 (Spherical) k-means 를 이용한 문서 군집화 학습 결과를 시각화합니다.

### Spherical k-means (soyclustering)

문서 군집화를 위해서는 Euclidean distance 가 아닌 Cosine distance 를 이용하는 Spherical k-means 을 이용해야 합니다. soyclustering 은 Cosine distance 용 fast initializer 와 clustering labeling 기능 및 Spherical k-means 을 제공합니다. 'similar_cut' 은 Cosine distance 용 initializer 입니다.

```python
from soyclustering import SphericalKMeans
from soyclustering import proportion_keywords

# spherical k-means
spherical_kmeans = SphericalKMeans(n_clusters=1000, max_iter=10, verbose=1, init='similar_cut‘)
labels = spherical_kmeans.fit_predict(x)

# clustering labeling
vocabs = ['this', 'is', 'vocab', 'list']
centers = kmeans.cluster_centers_
keywords = proportion_keywords(centers, labels, vocabs)
```

### k-means + pyLDAvis

Spherical k-means 의 학습 결과를 pyLDAvis 를 이용하여 시각화 합니다.

LDAvis 는 두 종류의 기준을 이용하여 topic keywords 를 선정합니다. P(w|t) 는 topic t 에서 word w 가 발생할 확률입니다. 이 값은 각 topic 에 dominant words 가 무엇인지를 표현합니다. 하지만 모든 topics 에서 흔하게 등장하는 단어는 P(w|t) 가 큽니다. 이를 보정하기 위하여 lift, P(w|t) / P(w) 도 이용합니다. 그리고 이 두 기준은 lambda 에 의하여 가중평균되어 최종 topic keyword score 인 relevance(w, t, lambda) 가 계산됩니다.

relevance(w, t, lambda) = lambda * P(w|t) + (1 - lambda) * P(w|t) / P(w)

Spherical k-means 의 학습 결과 우리는 cluster centers 와 labels 를 얻을 수 있습니다. soyclustering 의 keywords score 는 한 단어가 속한 군집에서의 등장 비율이 다른 군집에서의 등장 비율보다 얼마나 높은지, 그 비율을 keyword score 로 이용합니다. 이를 pyLDAvis 의 P(w|t) / P(w) 로 대체합니다. 그리고 centroid vector 의 weight 로 P(w|t) 를 대체합니다. 

또한 LDAvis 는 topic - term distribution 에 PCA 를 적용하여 2 차원으로 축소된 topic coordinates 를 얻습니다. k-means 의 학습 결과로 얻어진 centroid vectors 를 2 차원으로 축소하여 cluster coordinates 를 얻습니다. 하지만, PCA 는 비슷한 두 벡터를 거의 같은 2 차원 벡터로 압축하여 점이 겹치게 보입니다. PCA 대신 t-SNE 를 이용할 수도 있습니다. kmeans_to_pyLDAvis 에서는 PCA 와 t-SNE 를 선택할 수 있습니다.

kmeans_to_prepared_data 함수는 이 과정을 거쳐 pyLDAvis 용 PreparedData 를 만듭니다.

```python
# k-means + pyLDAvis
import pyLDAvis
from kmeans_visualizer import kmeans_to_prepared_data

prepared_data = kmeans_to_prepared_data(
    x, index2word, centers, labels,
    embedding_method='tsne'
)
pyLDAvis.display(prepared_data)
```

pyLDAvis 를 이용하여 만들어진 prepared_data 를 display 하면 pyLDAvis 와 같은 그림을 얻을 수 있습니다.

![](https://github.com/lovit/kmeans_to_pyLDAvis/blob/master/assets/kmeans_to_pyldavis_snapshot.png)


## Requires

- pandas >= 0.23.4
- pyLDAvis >= 2.1.1
- soyclustering : [github.com/lovit/clustering4docs](github.com/lovit/clustering4docs)