## Day 4 tutorials

### Hidden Markov Model (HMM) 을 이용한 한국어 형태소 분석기

day4_hmm_tagger 에서 다룹니다.

### 연관어 / 키워드 추출

각 문서 집합에서의 단어 분포 비율을 이용하면 각 문서 집합의 키워드나 한 단어의 연관어를 찾을 수 있습니다. `day_4_2_proportion_ratio_keyword` 에서는 단어 분포 비율을 이용한 키워드 / 연관어 추출기를 구현합니다.

Coverage 가 높으면서도 discriminative power 가 큰 단어들은 키워드로 적합합니다. `day_4_3_keyword_extraction_using_lasso_regression` 에서는 Lasso Regression 을 이용하여 단어의 연관어와 문서 집합의 키워드를 추출합니다.

`day_4_a_keyword_extraction_using_soykeyword` 에서는 위의 두 과정을 편하게 이용할 수 있도록 몇 가지 기능을 넣어둔 `soykeyword` 의 사용법을 알아봅니다.

### n-gram extraction

Point Mutual Information (PMI) 을 이용하면 bigram score 를 계산할 수 있습니다. 그리고 (w0, w1), (w1, w2) 에 대한 PMI 를 계산하면 trigram score 도 계산할 수 있습니다. `day_4_4_customized_ngram` 에서는 ngram extractor 를 직접 구현하고, 이를 이용하는 customized tokenizer 를 만든 뒤, scikit-learn 의 CountVectorizer 에 입력하여 term frequency matrix 를 만들어 봅니다.


## Requirements

이 실습 코드에서는 아래의 외부 패키지를 이용합니다.

1. [lovit_textmining_dataset](https://github.com/lovit/textmining_dataset)

```
git clone https://github.com/lovit/textmining_dataset
```

2. [Scikit-learn](https://scikit-learn.org/)

```
pip install scikit-learn
```

3. [soykeyword >= 0.0.14](https://github.com/lovit/soykeyword/)

```
pip install soykeyword
git clone https://github.com/lovit/soykeyword
```
