# MNIST에서 LLM으로 넘어가기 위해 반드시 알아야 할 개념

이 문서는 MNIST 손글씨 숫자 인식 과제에서 배운 내용 중, 다음 과제인 mini GPT/LLM 구현에 직접 이어지는 개념만 정리한 문서입니다.

MNIST의 목표는 이미지를 보고 0~9 중 하나를 맞히는 것입니다.

LLM의 목표는 문맥을 보고 다음 토큰을 맞히는 것입니다.

겉으로는 이미지 분류와 언어 모델이 달라 보이지만, 내부 학습 흐름은 매우 비슷합니다.

```text
입력
-> 숫자 벡터로 표현
-> 여러 Linear 계산
-> 확률 계산
-> 정답과 비교해 loss 계산
-> 역전파
-> optimizer로 파라미터 업데이트
```

## 1. 입력을 숫자로 바꾸는 과정

MNIST에서는 이미 이미지가 숫자로 바뀌어 있습니다.

```text
28 x 28 이미지
-> 784개 픽셀값
-> (batch_size, 784)
```

LLM에서는 문장을 바로 계산할 수 없습니다.

그래서 먼저 토크나이저가 필요합니다.

```text
문장
-> 토큰
-> 토큰 ID
-> (batch_size, sequence_length)
```

대응 관계는 다음과 같습니다.

| MNIST | LLM |
| --- | --- |
| 이미지 픽셀값 | 토큰 ID |
| `x_train` | tokenized text |
| `(batch, 784)` | `(batch, seq_len)` |
| 숫자 0~9 분류 | 다음 토큰 분류 |

**핵심:** 신경망은 문자나 이미지를 직접 이해하는 것이 아니라, 숫자 배열을 입력으로 받아 계산합니다.

## 2. Shape 감각

MNIST에서 가장 중요한 shape는 다음입니다.

```text
x: (batch_size, 784)
W: (784, hidden_dim)
out: (batch_size, hidden_dim)
```

LLM에서는 shape가 한 차원 더 늘어납니다.

```text
token_ids: (batch_size, seq_len)
embedding: (batch_size, seq_len, embed_dim)
logits: (batch_size, seq_len, vocab_size)
```

MNIST에서 shape를 이해하지 못하면 LLM에서 Attention의 Q, K, V shape를 이해하기 어렵습니다.

LLM에서 자주 나오는 shape:

```text
Q: (batch, heads, seq_len, head_dim)
K: (batch, heads, seq_len, head_dim)
V: (batch, heads, seq_len, head_dim)
attention_score: (batch, heads, seq_len, seq_len)
```

**핵심:** MNIST의 `x @ W`는 LLM의 Linear projection을 이해하기 위한 가장 작은 연습입니다.

## 3. Affine 계층은 LLM의 Linear 계층이다

MNIST에서 Affine 계층은 다음 계산을 합니다.

```text
out = xW + b
```

LLM에서는 이것을 보통 Linear layer라고 부릅니다.

Transformer 안에서도 Linear 계산이 계속 나옵니다.

```text
Embedding vector
-> Q projection
-> K projection
-> V projection
-> output projection
-> feed forward network
-> vocab projection
```

모두 본질은 다음과 같습니다.

```text
out = xW + b
```

MNIST 과제에서 Affine backward를 이해해야 하는 이유:

```text
dx = dout @ W.T
dW = x.T @ dout
db = sum(dout, axis=0)
```

LLM에서도 모든 Linear layer의 파라미터는 역전파로 업데이트됩니다.

**핵심:** LLM은 거대한 Linear 계산들이 Attention 구조와 함께 쌓인 모델입니다.

## 4. 활성화 함수

MNIST에서는 은닉층에 ReLU를 사용했습니다.

```text
ReLU(x) = max(0, x)
```

LLM에서는 ReLU보다 GELU가 자주 사용됩니다.

```text
GELU(x)
```

둘 다 하는 일은 비슷합니다.

```text
Linear 계산만으로는 표현력이 부족하므로
중간에 비선형성을 넣어 더 복잡한 패턴을 배우게 한다.
```

비교:

| 함수 | 주 사용처 | 특징 |
| --- | --- | --- |
| ReLU | 기본 MLP, MNIST | 단순하고 빠름 |
| Sigmoid | 예전 신경망, 확률 해석 | gradient가 사라지기 쉬움 |
| GELU | Transformer, GPT | 부드러운 비선형 함수 |
| Softmax | 출력 확률, Attention | 합이 1인 확률 분포 생성 |

**핵심:** LLM에서도 Linear 뒤에는 비선형 함수가 필요합니다. GPT의 FFN에서는 보통 GELU가 그 역할을 합니다.

## 5. Softmax

MNIST에서 Softmax는 10개 숫자 클래스의 확률을 만듭니다.

```text
logits: (batch, 10)
softmax(logits): (batch, 10)
```

LLM에서는 vocab 전체에 대해 다음 토큰 확률을 만듭니다.

```text
logits: (batch, seq_len, vocab_size)
softmax(logits): (batch, seq_len, vocab_size)
```

MNIST:

```text
이 이미지는 0~9 중 무엇인가?
```

LLM:

```text
이 위치의 다음 토큰은 vocab 중 무엇인가?
```

또 하나 중요한 점은 Attention에서도 Softmax가 사용된다는 것입니다.

```text
attention_score = Q @ K.T
attention_weight = softmax(attention_score)
```

Softmax의 역할:

```text
점수들을 확률처럼 바꾼다.
큰 점수는 더 큰 확률이 된다.
전체 합은 1이 된다.
```

**핵심:** Softmax는 MNIST 출력층뿐 아니라 LLM의 다음 토큰 예측과 Attention 가중치 계산에도 사용됩니다.

## 6. Cross Entropy Loss

MNIST에서는 정답 숫자에 해당하는 확률을 보고 loss를 계산합니다.

```text
정답이 7이면
모델이 7에 준 확률만 뽑아서 loss 계산
```

LLM에서도 똑같습니다.

```text
정답 다음 토큰이 "밥"이면
모델이 "밥" 토큰에 준 확률로 loss 계산
```

수식의 느낌:

```text
loss = -log(정답 클래스 확률)
```

batch 평균:

```text
loss = -mean(log(정답 확률))
```

MNIST와 LLM의 차이:

| 구분 | MNIST | LLM |
| --- | --- | --- |
| 정답 | 숫자 라벨 0~9 | 다음 토큰 ID |
| 클래스 수 | 10 | vocab_size |
| loss 대상 | 이미지 1개당 정답 1개 | 각 token 위치마다 다음 토큰 |

**핵심:** LLM 학습도 결국 "정답 토큰 확률을 높이는 방향"으로 Cross Entropy Loss를 줄이는 과정입니다.

## 7. Forward, Backward, Update

MNIST 학습 루프:

```text
1. forward
2. loss
3. backward
4. optimizer update
```

LLM 학습 루프도 같습니다.

```text
1. token 입력
2. embedding
3. transformer forward
4. logits
5. cross entropy loss
6. backward
7. optimizer update
```

MNIST에서 꼭 이해해야 할 점:

```text
forward는 예측값을 만드는 과정
backward는 loss 기준으로 각 파라미터의 gradient를 구하는 과정
update는 gradient를 이용해 파라미터를 바꾸는 과정
```

LLM에서도 학습되는 파라미터:

```text
embedding table
Q, K, V projection weight
FFN weight
LayerNorm gamma/beta
output projection weight
```

**핵심:** 모델이 커져도 학습 원리는 `loss를 줄이도록 gradient로 파라미터를 업데이트한다`입니다.

## 8. Optimizer: SGD와 Adam

MNIST에서는 SGD와 Adam을 비교했습니다.

SGD:

```text
params -= lr * grads
```

Adam:

```text
gradient 평균 m
gradient 제곱 평균 v
보정 후 업데이트
```

LLM에서는 보통 Adam 계열을 사용합니다.

이유:

```text
파라미터가 매우 많고
gradient 크기가 위치마다 다르며
안정적인 학습이 중요하기 때문
```

MNIST에서 Adam을 이해해야 LLM의 optimizer 설정을 이해할 수 있습니다.

자주 나오는 하이퍼파라미터:

```text
learning rate
beta1
beta2
weight decay
warmup
```

**핵심:** LLM은 대개 SGD보다 Adam/AdamW 같은 optimizer를 사용합니다.

## 9. Batch 학습

MNIST에서는 여러 이미지를 batch로 묶어서 학습합니다.

```text
x_batch: (batch_size, 784)
y_batch: (batch_size,)
```

LLM에서는 여러 문장 또는 토큰 조각을 batch로 묶습니다.

```text
input_ids: (batch_size, seq_len)
target_ids: (batch_size, seq_len)
```

batch를 쓰는 이유:

```text
계산을 효율적으로 하기 위해
gradient를 더 안정적으로 만들기 위해
GPU 연산을 잘 활용하기 위해
```

**핵심:** LLM 학습도 batch 단위로 forward, loss, backward, update를 반복합니다.

## 10. 초기화: He와 Xavier

MNIST 과제에서는 ReLU를 사용했기 때문에 He 초기화를 사용했습니다.

```text
He: sqrt(2 / fan_in)
```

Xavier:

```text
Xavier: sqrt(1 / fan_in)
```

차이:

```text
He는 분산 기준으로 Xavier의 2배
표준편차 기준으로는 sqrt(2)배
```

LLM에서도 초기화는 중요합니다.

모델이 깊어질수록 값이 너무 커지거나 작아지면 학습이 불안정해집니다.

**핵심:** 초기화는 forward 값과 backward gradient가 층을 지나며 망가지지 않게 하는 출발점입니다.

## 11. BatchNorm과 LayerNorm

MNIST에서는 BatchNorm을 배웠습니다.

BatchNorm:

```text
batch 안에서 같은 feature 위치끼리 평균/분산 계산
```

LLM에서는 보통 BatchNorm보다 LayerNorm을 씁니다.

LayerNorm:

```text
각 token vector 내부 feature들 기준으로 평균/분산 계산
```

왜 LLM은 LayerNorm을 쓰는가?

```text
문장 길이가 다양하고
batch 통계에 의존하면 불안정할 수 있으며
sequence 모델에서는 token별 정규화가 더 자연스럽기 때문
```

대응 관계:

| MNIST | LLM |
| --- | --- |
| BatchNorm | LayerNorm |
| batch 기준 정규화 | token vector 기준 정규화 |
| 학습/추론 동작 차이 있음 | 학습/추론 동작 차이가 상대적으로 단순 |

**핵심:** MNIST에서 BatchNorm을 완벽히 외우기보다, "중간값 분포를 안정화해 학습을 돕는다"는 목적을 이해하면 LLM의 LayerNorm으로 넘어가기 쉽습니다.

## 12. Dropout

MNIST에서는 Dropout으로 일부 뉴런을 랜덤하게 껐습니다.

```text
학습 중 일부 출력을 0으로 만들어 과적합을 줄임
```

LLM에서도 Dropout이 사용될 수 있습니다.

사용 위치 예:

```text
embedding dropout
attention dropout
residual dropout
FFN dropout
```

역할은 같습니다.

```text
모델이 특정 경로에 너무 의존하지 않도록 함
과적합을 줄임
```

**핵심:** Dropout은 정확도를 항상 바로 올리는 기능이 아니라, 일반화를 돕는 규제 기법입니다.

## 13. MNIST와 LLM의 가장 중요한 대응표

| MNIST 개념 | LLM 개념 | 왜 중요한가 |
| --- | --- | --- |
| 이미지 픽셀 | 토큰 ID | 입력을 숫자로 바꾸는 단계 |
| Affine | Linear layer | 모든 큰 계산의 기본 |
| ReLU | GELU | 비선형성 부여 |
| Softmax | token probability, attention weight | 확률 분포 생성 |
| Cross Entropy | next token prediction loss | 정답 토큰 확률 학습 |
| SGD/Adam | Adam/AdamW | 파라미터 업데이트 |
| BatchNorm | LayerNorm | 중간값 안정화 |
| Dropout | Transformer Dropout | 과적합 완화 |
| batch | batch + sequence | 효율적인 학습 단위 |
| train/evaluate | train/generate/evaluate | 학습과 사용 모드 구분 |

## 14. LLM으로 넘어가기 전 체크리스트

다음 질문에 답할 수 있으면 LLM 구현으로 넘어갈 준비가 된 것입니다.

1. `x @ W + b`의 shape를 설명할 수 있는가?
2. Softmax가 왜 확률처럼 해석되는지 설명할 수 있는가?
3. Cross Entropy가 왜 정답 클래스 확률만 보는지 설명할 수 있는가?
4. 역전파가 왜 `dW`, `db`를 구하는 과정인지 설명할 수 있는가?
5. Adam이 SGD와 어떻게 다른지 설명할 수 있는가?
6. batch 단위 학습이 왜 필요한지 설명할 수 있는가?
7. train과 inference의 차이를 설명할 수 있는가?
8. BatchNorm의 목적을 LayerNorm과 연결해서 말할 수 있는가?
9. Dropout이 정확도 상승 도구가 아니라 규제 도구라는 점을 설명할 수 있는가?
10. MNIST의 숫자 분류와 LLM의 다음 토큰 예측이 둘 다 다중분류 문제라는 점을 설명할 수 있는가?

## 15. LLM 구현으로 이어지는 최종 흐름

MNIST:

```text
image pixels
-> Affine
-> ReLU
-> Affine
-> Softmax
-> Cross Entropy
-> Backward
-> Adam
```

LLM:

```text
text
-> tokenizer
-> token IDs
-> embedding
-> Transformer blocks
-> Linear vocab head
-> Softmax
-> Cross Entropy
-> Backward
-> Adam
```

LLM에서 새로 배울 것:

```text
tokenizer
embedding
positional encoding
self-attention
causal mask
multi-head attention
layer normalization
GELU
residual connection
decoding
```

하지만 그 아래에서 계속 반복되는 기본기는 MNIST와 같습니다.

```text
행렬곱
확률
loss
gradient
optimizer
batch 학습
```

## 16. 한 문장 요약

MNIST에서 반드시 가져가야 할 것은 숫자 배열을 입력으로 받아 Linear 계산을 쌓고, Softmax와 Cross Entropy로 정답 확률을 학습하며, 역전파와 Adam으로 파라미터를 업데이트하는 전체 흐름입니다. 이 흐름이 LLM에서는 이미지 대신 토큰, Affine 대신 Transformer 내부 Linear, 숫자 정답 대신 다음 토큰 정답으로 확장됩니다.
