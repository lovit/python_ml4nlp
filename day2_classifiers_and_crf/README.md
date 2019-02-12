## Day 2 tutorials

Day1 에서는 기본적인 classifiers 들과, sequential labeling algorithm 인 Conditional Random Field 의 사용법을 연습합니다.

### Classifiers

벡터 형식으로 입력된 문장이 긍정인지 부정인지를 판단하는 classifications 에는 다양한 알고리즘들이 이용될 수 있습니다.

`day_2_0_classifiers_usage` 에서는 scikit-learn 을 이용하여 아래의 알고리즘들의 사용법을 연습합니다.

- Logistic Regression
- Feed forward Neural Network
- Support Vector Machine (linear kernel, Radious Basis Function kernel)
- Naive Bayes
- Decision Tree

`day_2_1_classification_comparison` 에서는 Decision Tree 와 Logistic Regression 의 parameters 별 모델의 분류 성능과 복잡도 (complexity) 를 비교합니다. 이를 위하여 scikit-learn 에서 제공하는 cross validation 기능을 이용합니다.

### Conditional Random Field and Space correction

Conditional Random Field (CRF) 는 sparse representation 을 이용하는 sequential labeling 방법들 중에서는 여러 문제에서 안정적이고 좋은 성능을 보여준다고 알려져 있습니다.

그리고 한국어의 띄어쓰기 교정 문제는 character sequence 에 각각 `띈다`, `안띈다` 의 label 을 부여하는 sequential labeling 문제로 볼 수 있습니다.

`day_2_2_spacing_correction_using_crf` 에서는 python-crfsuite 패키지를 이용하여 영화 라라랜드 리뷰 띄어쓰기 교정기를 만들어 봅니다.

## Requirements

이 실습 코드에서는 아래의 외부 패키지를 이용합니다.

1. [lovit_textmining_dataset](https://github.com/lovit/textmining_dataset)

```
git clone https://github.com/lovit/textmining_dataset
```

2. [python_crfsuite](https://github.com/scrapinghub/python-crfsuite)

```
pip install python-crfsuite
```

3. [pycrfsuite_spacing >= 1.0.0](https://github.com/lovit/pycrfsuite_spacing)

```
pip install pycrfsuite_spacing
```

4. [Scikit-learn](https://scikit-learn.org/)

```
pip install scikit-learn
```
