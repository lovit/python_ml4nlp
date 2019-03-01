## Graph ranking

Graph 에서 중요한 마디를 찾기 위하여 PageRank 나 HITS 와 같은 graph ranking algorithms 이 이용됩니다. TextRank 는 Word adjacent graph 를 만든 뒤, PageRank 를 적용하여 중요한 키워드를 찾습니다. Sentence similarity 를 이용하여 sentence graph 를 만든 뒤 PageRank 를 적용하면 key sentences 를 추출할 수 있습니다.

`day_8_0_text_to_graph_utils` 에서는 word adjacent graph 나 document - term frequency matrix 를 graph 로 만드는 방법과 함수들을 알아봅니다.

`day_8_1_pagerank_and_textrank` 에서는 PageRank 를 직접 구현하고, 이를 이용한 TextRank 도 구현합니다.

`day_8_2_word_cloud_with_kr-wordrank` 에서는 KR-WordRank 를 이용하여 토크나이저를 이용하지 않으면서 homogeneous dataset 에서 키워드를 추출합니다.

`day_8_3_textrank_lovit` 와 `day_8_4_gensim_textrank` 는 모두 TextRank 를 구현한 구현체 입니다. 특히 8-4 에서는 gensim 에서 제공하는 summarizer 에 대하여 알아봅니다. Gensim 의 구현체는 TextRank 의 변형인 LexRank 입니다.


## Graph similarity

Graph 에서 이웃 간의 관계가 비슷한 마디를 찾기 위하여 SimRank 나 Random Walk with Restart 와 같은 graph similarity (ranking) algorithm 이 이용됩니다. `day_8_5_simrank_and_rwr` 는 SimRank 와 RWR 의 구현체를 이용하여 topically similar words 를 탐색합니다.


## Requirements

이 실습 코드에서는 아래의 외부 패키지를 이용합니다.

1. [KR-WordRank](https://github.com/lovit/kr-wordrank)

```
pip install krwordrank
```

2. [soygraph](https://github.com/lovit/soygraph)

```
git clone https://github.com/lovit/soygraph
```

3. [TextRank](https://github.com/lovit/textrank)

```
git clone https://github.com/lovit/textrank
```

4. [WordCloud](https://github.com/lovit/textrank)

```
pip install wordcloud
```
