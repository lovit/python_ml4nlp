## PyTorch basic

PyTorch 로 모델을 만들 때에는 (1) data, (2) model, (3) loss function, (4) optimizer 를 잘 설정하면 됩니다. 

`day_8_pytorch_basic_and_regression` 에서는 1 차원 곡선 데이터를 예측하는 non linear regression model 을 만들어 봄으로써 위 네가지 요소를 구현하는 연습을 합니다. 또한 nn.Module 을 상속한 모델을 만들 때에 nn.Sequential 을 이용하는 방식과 forward 함수에 각 layer 의 계산 과정을 직접 구현하는 연습도 합니다.

`day_8_softmax_regression_for_document_classification` 에서는 scikit-learn 에서 제공하는 20 news group dataset 을 이용하여 문서 분류를 하는 softmax regression 을 구현합니다. PyTorch 에서는 COO matrix 형식의 sparse matrix 를 제공합니다 ([링크][sparse]). 이번 튜토리얼에서는 scipy.sparse.csr_matrix 에서 sub matrix 를 만든 뒤, 이를 dense matrix 로 변환하는 형식으로 구현하였습니다.

[sparse]: https://pytorch.org/docs/stable/sparse.html

## Requirements

1. PyTorch

PyTorch 는 각 OS 와 Python 의 버전, GPU 사용 유무에 따라 설치 스크립트가 다릅니다. 이는 [공식 홈페이지][pytorch_official]에 소개되어 있습니다.

[pytorch_official]: https://pytorch.org/
