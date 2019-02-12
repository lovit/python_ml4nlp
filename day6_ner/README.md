## Conditional Random Field 를 이용한 Named Entity Recognition

Word embedding 이 Named Entity Recognition 에 이용되기 전에는 Conditional Random Field 이 주로 이용되었습니다.

`day_6_CoNLL2002_NER_pycrfsuite.ipynb` CoNLL 2002 의 NER task dataset 을 이용하여 NER 을 학습합니다. 또한 앞/뒤의 단어 만으로도 충분히 Named Entity 가 잘 추출됨을 확인합니다.

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
