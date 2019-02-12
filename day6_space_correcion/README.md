## soyspacing 을 이용한 띄어쓰기 교정기 학습

한국어의 띄어쓰기 교정 문제는 **본래 띄어써야 하는데 붙여쓴 경우를 교정하는 것**입니다. 

`day_6_a_soyspacing.ipynb` 에서는 CRF 의 단점을 보완하여 보수적인 띄어쓰기 교정을 하는 soyspacing package 의 사용법을 연습합니다.

## Requirements

이 실습 코드에서는 아래의 외부 패키지를 이용합니다.

1. [lovit_textmining_dataset](https://github.com/lovit/textmining_dataset)

```
git clone https://github.com/lovit/textmining_dataset
```

2. [soyspacing == 1.0.15](https://github.com/lovit/soyspacing)

```
pip install soyspacing
```