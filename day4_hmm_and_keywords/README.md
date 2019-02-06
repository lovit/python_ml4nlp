## Day 4 tutorials

### Hidden Markov Model (HMM) 을 이용한 한국어 형태소 분석기

Hidden Markov Model (HMM) 은 초기의 형태소 분석기에서도 이용되던 sequential labeling 방법입니다. 그리고 HMM 을 이용하여 탐색되는 최적의 label sequences 는  word adjacent graph 에서의 shortest path 이기도 합니다.

`day_4_0_shortest_path` 에서는 Dijkstra algorithm 을 이용하여 지하철 역 간의 최단경로를 찾는 연습을 합니다.

`day_4_1_hmm_postagger` 에서는 단어 사전과 미리 학습된 용언 분석기 (lemmatizer) 를 이용하여 문장에서 word adjacent graph 를 만들고, 앞서 연습한 Dijkstra algorithm 을 이용하여 주어진 문장에서 최적의 형태소열을 찾습니다. 파이썬만을 이용하여 이 과정을 직접 구현합니다.

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

2. [korean_lemmatizer](https://github.com/lovit/korean_lemmatizer)

학습데이터를 이용하는 한국어 용언 분석기 (형태소 분석기) 입니다. Hidden Markov Model (HMM) 기반 한국어 형태소 분석기의 구현 예제 (4-1) 에서 이용합니다.

```
git clone https://github.com/lovit/korean_lemmatizer
```

3. [Scikit-learn](https://scikit-learn.org/)

```
pip install scikit-learn
```

4. [soykeyword >= 0.0.14](https://github.com/lovit/soykeyword/)

```
pip install soykeyword
git clone https://github.com/lovit/soykeyword
```

## Appendix

1. [shortestpath](https://github.com/lovit/shortestpath)

Shortest path 를 위한 dijkstra algorithm 구현체 입니다. Repository 에 더 많은 예제가 있습니다.

```
git clone https://github.com/lovit/shortestpath
```
