## String distance

단어나 문장의 형태적 유사성을 정의하기 위하여 string distance 가 이용될 수 있습니다. Levenshtein distance (edit distance) 는 대표적인 string distance 입니다. Levenshtein distance 를 한글에 적용할 때에는 초/중/종성을 분리하면 더 정밀한 거리 정의가 가능합니다. `day_7_string_distance.ipynb` 에서는 한글에 대하여 단어 수준 / 초중종성 수준 / 어절 수준에 대한 Levenshtein distance 를 직접 구현합니다. 그 외에도 Jaccard distance, Cosine distance 도 구현합니다.

단어는 character level sparse vector 로 생각할 수 있습니다. Sparse vector 형태로 표현된 데이터에서 k-nearest neighbors 를 찾을 때에는 inverted index 도 유용합니다. `day_7_hangle_levenshtein_with_inverted_index.ipynb` 에서는 inverted index 를 이용하여 효율적으로 형태가 유사한 단어를 검색합니다.


## Requirements

이 실습 코드에서는 아래의 외부 패키지를 이용합니다.

1. [Inverted index for hangle edit distance](https://github.com/lovit/inverted_index_for_hangle_editdistance)

```
git clone https://github.com/textmining_dataset
```
