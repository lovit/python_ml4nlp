## Day 1 tutorials

Day1 에서는 KoNLPy 를 이용하여 문서집합을 term frequency matrix (Bag of Words Model) 로 만들고, 이를 이용하여 영화 평을 긍정과 부정으로 분류하는 logistic regression 을 학습합니다.

###  Create and import  Python package

데이터 분석 작업을 하다보면 반복적인 작업에 대한 각자의 함수가 쌓입니다. 이들을 잘 정리하여 패키지로 만들어두면 이후에 코드를 재활용하여 효율적인 분석이 가능합니다.

`day_1_0_create_and_import_packages` 에서는 함수를 저장한 py 파일을 만들고, 외부 폴더에 py 파일들을 저장하였을 경우, 이를 불러들이는 연습을 합니다.

### KoNLPy

KoNLPy 는 파이썬 환경에서 다양한 종류의 한국어 형태소 분석기 / 한국어 품사 판별기를 이용할 수 있도로고 도와주는 패키지입니다.

`day_1_1_KoNLPy` 이를 이용하여 문서를 단어열로 만들고, 단어의 빈도수를 계산하는 연습을 합니다. 또한 `KoNLPy >= 0.5.0` 이후에 `Komoran` 에 추가된 사용자 사전 기능을 이용하는 방법도 연습합니다.

### From text to sparse matrix

텍스트 데이터를 term frequency matrix 로 표현하기 위해서는 tokenization, stopword filtering, vectorizing 과정을 거칩니다. 이를 위하여 여러 토크나이저들을 이용할 수 있고, 후처리 기능을 추가할 수도 있습니다.

`day_1_2_from_text_to_sparse_matrix` 에서는 텍스트 파일이 하나 주어졌을 때 KoNLPy 를 이용하여 이를 단어열로 만들고 후처리 과정을 거치는 custom tokenizer 를 만듭니다. 그리고 scikit-learn 에서 제공하는 CountVectorizer 에 이를 입력하여 사용하는 연습을 합니다. 학습된 sparse matrix 를 저장하고 불러 읽는 연습도 합니다.

### Logistic regression and L1, L2 regularization

day1_logistic_regression 폴더에서 다룹니다.


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
