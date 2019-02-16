# 빠른 한글 수정 거리 검색을 위한 inverted index

### Indexing

```python
from fast_hangle_levenshtein import LevenshteinIndex
indexer = LevenshteinIndex(verbose=True)

indexer.indexing('아이고 어이고 아이고야 아이고야야야야 어이구야 지화자 징화자 쟝화장'.split())
```

### Searching

Levenshtein distance

```python
indexer.levenshtein_search('아이코', max_distance=1)
```

Jamo Levenshtein distance

```python
indexer.jamo_levenshtein_search('아이코', max_distance=4/3)
```

### Performance

With 132,864 nouns,

```python
print(len(nouns)) # 132,864
```

```python
from fast_hangle_levenshtein import LevenshteinIndex

financial_word_indexer = LevenshteinIndex(nouns)
financial_word_indexer.verbose = True
financial_word_indexer.levenshtein_search('분식회계')
```

    query=분식회계, candidates=10137 -> 7, time=0.00606 sec.
    [('분식회계', 0), ('분식회', 1), ('분식회계설', 1), ('분석회계', 1)]

Using Levenshtein distance for all 132,864 reference words,

```python
import time
from fast_hangle_levenshtein import levenshtein
from fast_hangle_levenshtein import jamo_levenshtein

query = '분식회계'

search_time = time.time()
distance = {word:levenshtein(word, query) for word in noun_scores}
search_time = time.time() - search_time
print('search time = {} sec'.format('%.2f'%search_time)) # search time = 2.27 sec
```

With same results.

```python
similars = sorted(filter(lambda x:x[1] <= 1, distance.items()), key=lambda x:x[1])
print(similars) # [('분식회계', 0), ('분식회', 1), ('분식회계설', 1), ('분석회계', 1)]
```