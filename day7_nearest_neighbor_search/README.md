## Approximated Nearest Neighbor Search

k-nearest neighbors model 은 classification / prediction 이나 검색과 같은 작업에 이용될 수 있습니다. k-NN 을 찾기 위하여 database 의 모든 reference 와 query 의 거리를 계산하지는 않습니다. 정확성을 조금 잃더라도 빠른 시간에 k-NN (혹은 그와 유사한) 을 찾는 문제를 approximated nearest neighbor search 라 합니다. 그리고 이 문제에 가장 널리 이용되는 방법은 random projection 을 이용하는 Locality Sensitive Hashing 입니다.

`day_7_sklearn_lsh_usage.ipynb` 에서는 random samples 을 만들어 LSHForest 의 사용법을 살펴봅니다.

`day_7_find_similar_documents.ipynb` 에서는 한 뉴스와 topically similar 한 뉴스를 찾는 연습을 합니다.


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
