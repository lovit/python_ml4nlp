## Day 3 tutorials

Day 3 에서는 통계 기법을 이용하여 단어와 명사를 추출하는 unsupervised word extraction 방법들에 대하여 알아봅니다 그리고 이를 이용하는 unsupervised tokenizer 에 대해서도 알아봅니다.

Word Piece Model (WPM) 은 sentence piece 라는 이름으로도 이용되며, Recurrent Neural Network 와 같이 vocabulary size 가 한정적인 상황에서 이용할 수 있는 unsupervised tokenizer 입니다. 이는 단어를 subwords 로 표현합니다.

`day_3_0_navernews_dataset` 에서는 `lovit_textmining_dataset` 에서 `navernews_10days` 데이터셋을 불러들이고 `soynlp` 의 `DoublespaceLineCorpus` 를 이용하여 문장과 문서 단위로 데이터를 받는 연습을 합니다.

### Unsupervised Word Extraction & Tokenization

Unsupervised word extraction 을 위하여 Cohesion score, Branching Entropy, Accessor Variety 등이 이용될 수 있습니다. `day_3_1_cohesion_branching_entropy` 에서는 이들을 학습하는 코드를 직접 만들어 봅니다.

`day_3_3_noun_extraction` 에서는 한국어 어절의 구조 (L + [R]) 를 이용하는 명사 추출 방법을 구현합니다.

`day_3_4_word_noun_tokenizer_soynlp` 에서는 앞의 코드들에 몇 가지 기능을 더하여 한국어 단어 추출을 위해 구현된 `soynlp` 패키지의 사용법을 연습합니다.

### Word Piece Model

`day_3_2_WordPieceModel` 에서는 원 논문에서 제공하는 코드를 이용하여 WPM 을 학습합니다. 하루치 뉴스 기사에 적용해 봄으로써 WPM 이 학습하는 subwords 들에 대해서도 살펴봅니다.

`day_3_2_sentencepiece_(google_package)` 에서는 Google 에서 공개한 코드를 이용하여 WPM 을 학습합니다. Sentencepiece 라는 패키지 이름으로 공개되었습니다. 계산 과정이 최적화되어 있고, C++ 로 구현되어 있기 때문에 실제로 사용할 때에는 이를 이용하면 좋습니다.

## Requirements

이 실습 코드에서는 아래의 외부 패키지를 이용합니다.

1. [lovit_textmining_dataset](https://github.com/lovit/textmining_dataset)

```
git clone https://github.com/lovit/textmining_dataset
```

2. [soynlp](https://github.com/lovit/soynlp/)

```
pip install scikit-learn

git clone https://github.com/lovit/soynlp/
```
