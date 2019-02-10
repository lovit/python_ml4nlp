## Day 1 tutorials

Day1 에서는 KoNLPy 를 이용하여 문서집합을 term frequency matrix (Bag of Words Model) 로 만들고, 이를 이용하여 영화 평을 긍정과 부정으로 분류하는 logistic regression 을 학습합니다.

### Logistic regression and L1, L2 regularization

Logistic regression 은 대표적인 linear classifier 알고리즘입니다. 그리고 이 모델은 overfitting 을 방지하기 위하여 L1, L2 regularization 을 적용할 수도 있습니다.

`day_1_3_movie_sentiment_classification` 에서는 영화 평과 영화 평점을 이용하여 한 영화 평이 주어졌을 때 이의 긍/부정 여부를 판단하는 classifiers 를 학습합니다.


## Requirements

이 실습 코드에서는 아래의 외부 패키지를 이용합니다.

1. [lovit_textmining_dataset](https://github.com/lovit/textmining_dataset)

```
git clone https://github.com/lovit/textmining_dataset
```

2. [konlpy >= 0.5.1](https://github.com/konlpy/konlpy/)

```
pip install konlpy
```

3. [Scikit-learn](https://scikit-learn.org/)

```
pip install scikit-learn
```
