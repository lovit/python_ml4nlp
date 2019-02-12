## soyspacing 을 이용한 띄어쓰기 교정기 학습

한국어의 띄어쓰기 교정 문제는 **본래 띄어써야 하는데 붙여쓴 경우를 교정하는 것**입니다. 띄어쓰기 교정기가 이용하는 정보는 한 글자의 앞/뒤에 등장하는 다른 글자들입니다. Conditional Random Field (CRF) 는 potential function 을 이용하여 앞/뒤에 등장하는 글자들의 정보를 features 로 이용합니다. 그러나 CRF 는 모든 경우에 동일한 features 를 이용하는 단점이 있습니다. soyspacing 은 CRF 처럼 앞/뒤에 등장하는 글자들을 features 로 이용하여 띄어쓰기를 교정하지만, `이다음에`와 같은 상황에서는 `X[-1:0] = 이다`를 feature 로 이용하지 않으며 띄어쓰기를 교정합니다.

또한 CRF 는 softmax 기준으로 0.01 이라도 더 높은 쪽의 띄어쓰기 label 을 선택합니다. 하지만 0.49 : 0.51 과 같은 확률을 지니는 애매모호한 상황에서는 왠만하면 띄어쓰기를 하지 않는 편이 좋습니다. soyspacing 은 score threshold 를 두어, 특정 띄어쓰기 점수 이상이 될 때에만 띄어쓰기를 하며, 특정 붙여쓰기 점수 이하일 때에만 붙여쓰기를 합니다. 그 외에는 애매모호하기 때문에 '띄어쓰지 않기' 를 합니다.

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