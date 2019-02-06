## Python 기초

텍스트 데이터 분석을 위해 필요한 파이썬 기초 문법들을 다뤄봅니다. 

### Python dict, JSON

Table 형식의 structured data 가 아닌 경우에는 JSON 형태로 데이터를 저장할 경우가 많습니다. 종종 web scraping 의 결과를 JSON 으로 저장할 경우들이 있으며, API 를 통하여 전달받은 데이터의 형태가 JSON 인 경우들이 있습니다.

`day0_Python0_Dict_JSON_IO` 에서는 dict 를 다루는 방법과 이를 JSON 으로 저장하는 방법에 대하여 알아봅니다.

### Pickling

데이터를 사람이 읽을 수 있는 텍스트 형식으로 저장할 수도 있지만, 파이썬의 자료형을 그대로 유지하는 binary 형태로 저장할 수도 있습니다. pickle 은 파이썬에서 데이터를 binary 형태로 저장할 수 있도록 도와주는 패키지 입니다. 학습된 모델을 class instance 로 저장할 때에도 pickle 은 이용될 수 있습니다.

`day0_Python1_pickling` 에서는 pickle 의 사용법에 대하여 알아봅니다.

### List comprehension & enumerate

파이썬의 장점 중 하나는 두, 세 개의 for loop 을 한 줄로 작성할 수 있다는 점입니다. 예를 들어 아래 같은 코드는 맨 아래의 한 줄로 작업할 수 있습니다. 이는 list 뿐 아니라 dict, set 에도 동일하게 작동합니다.

```python
results = []
for a in A:
    for b in B:
        # do something
        results.append(result)

results = [# do something for a in A for b in B[
```

또 하나의 장점은 enumerate 함수입니다. Iterable 한 대상의 index 를 아래처럼 가져올 필요가 없습니다.

```python
list_a = [ ... ]
for i in range(len(list_a)):
    a = list_a[i]
    # do something
```

대신 아래와 같이 enumerate 를 이용하여 index 와 item 을 동시에 가져올 수 있습니다.

```python
for i, a in enumerate(list_a):
    # do something
```

`day0_Python2_list_comprehension_and_iteration` 에서는 list comprehension 문법을 연습합니다.

### sliceing and sorting

List 에서 sublist 를 가져오거나 str 에서 substr 을 가져올 수 있습니다. 그리고 파이썬에서 제공하는 sorted 함수를 이용하면 sort by value 와 같은 작업을 할 수 있습니다. 심지어 sort by function result 로도 작업을 할 수 있습니다.

`day0_Python3_slice_and_sorting` 에서는 파이썬에서 제공하는 slicing 과 sorting 작업을 연습합니다.

### yield

텍스트 데이터를 다룰 때에는 텍스트의 모든 데이터를 읽지 않고, 한 줄씩 (line by line) 작업을 처리할 때도 있습니다. 대표적인 예로 word counting 을 하려면, 한 줄에 포함된 단어의 빈도수를 counter 에 저장한 뒤, 해당 줄을 메모리에서 버리고 다음 줄을 로딩하여 작업을 반복하면 됩니다. 파이썬에서는 이와 같은 작업을 효율적으로 할 수 있도록 yield 기능을 제공합니다. `soynlp` 의 DoublespaceLineCorpus 도 yied 를 이용하는 데이터 클래스 입니다. 그 외에도 큰 용량의 데이터를 처리할 때 자주 이용될 기능입니다.

`day0_Python4_yield_len_iter` 에서는 yield 를 연습합니다. 이를 위하여 class 에 `__iter__` 함수를 구현해야 합니다. 또한 `__len__` 함수도 구현해 봄으로써, 파이썬 클래스의 기본 함수를 다루는 연습을 해봅니다.

### Word counting

텍스트 데이터 분석의 기본은 카운팅 입니다. 한 문서의 단어의 개수를 세는 방법은 다양합니다. 여러 방법에 대하여 연습을 함으로써 파이썬의 자료형과 패키지들에 익숙해 집니다.

`day0_Python5_three_way_for_word_counting` 에서는 `collections.Counter`, `collections.defaultdict`, 기본 자료형인 `dict` 를 이용하는 word counting 방법들을 연습합니다.

### zip, zip*

파이썬에서는 두 개 이상의 iterable 한 객체를 동시에 다룰 수 있습니다. 예를 들어 아래의 `l_a` 와 `l_b` 를 `l` 로 만들거나 그 반대의 과정을 손쉽게 할 수 있도록 zip 함수를 이용할 수 있습니다.

`day0_Python6_zip` 에서는 zip 함수를 이용하는 연습을 합니다.

```python
l = [('a', 0), ('b', 1), ('c', 2)]
l_a = ['a', 'b', 'c']
l_b = [0, 1, 2]
```

### Web scraping

웹에서 필요한 데이터를 수집할 수 있습니다. 이를 위해서는 HTML 에서 필요한 정보를 선택하는 연습을 해야 합니다.

`day0_Appendix0_scraping_naver_movie` 에서는 네이버 영화에서 영화 `라라랜드`의 정보들을 수집하는 예제와 이미지 파일을 다운로드 받는 연습을 합니다. User agent 개념과 headers 에 대해서도 알아봅니다.

### Handling excel

때로는 Microsoft Excel 형식의 데이터를 다룰 일도 있습니다. `day0_Appendix1_handling_excel-toy` 에서는 toy data 를 이용하여 한 엑셀 파일의 여러 sheets 에서 데이터를 가져오는 연습을 합니다.
