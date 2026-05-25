# SGD 실험 결과 정리

## 1. 실험 목적

이번 실험은 같은 신경망 구조에서 optimizer를 `SGD`로 고정하고, learning rate, 가중치 초기화 방식, BatchNorm/Dropout 적용 여부에 따라 학습 결과가 어떻게 달라지는지 확인하는 것을 목표로 한다.

활성화 함수는 모든 실험에서 `ReLU`를 사용했고, 출력층은 `Softmax`를 사용했다.

## 2. 실험 조건

| 번호 | Optimizer | BatchNorm | Dropout | Learning Rate | 초기화 | Test Accuracy | Total Params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SGD | False | False | 0.001 | He | 90.70% | 535,818 |
| 2 | SGD | False | False | 0.001 | Xavier | 90.76% | 535,818 |
| 3 | SGD | False | False | 0.01 | He | 95.77% | 535,818 |
| 4 | SGD | True | True | 0.001 | He | 87.49% | 537,354 |
| 5 | SGD | True | True | 0.001 | Random(no scaling) | 87.77% | 537,354 |

## 3. 실험별 관찰

### 3.1 SGD + He + lr=0.001

기본 설정으로 `BatchNorm`과 `Dropout`을 사용하지 않고, He 초기화와 SGD를 적용했다.

학습 loss는 꾸준히 감소했지만 최종 정확도는 `90.70%`로 목표 정확도인 95%에는 도달하지 못했다.

learning rate가 `0.001`로 작아서 학습이 안정적으로 진행되지만, 20 epoch 안에서 충분히 빠르게 수렴하지 못한 것으로 보인다.

![SGD, BatchNorm False, Dropout False, lr=0.001, He](test_img/sgd_1.png)

### 3.2 SGD + Xavier + lr=0.001

He 초기화 대신 Xavier 초기화를 적용했다.

정확도는 `90.76%`로 He 초기화 실험의 `90.70%`와 거의 차이가 없었다.

현재 모델은 깊이가 깊지 않고 MNIST 데이터셋이 비교적 단순하기 때문에, He와 Xavier 초기화 차이가 최종 정확도에서 크게 드러나지 않은 것으로 해석할 수 있다.

![SGD, BatchNorm False, Dropout False, lr=0.001, Xavier](test_img/sgd_2.png)

### 3.3 SGD + He + lr=0.01

learning rate를 `0.001`에서 `0.01`로 키웠다.

정확도는 `95.77%`로 크게 상승했고, loss도 훨씬 빠르게 감소했다.

이번 SGD 실험 중 가장 좋은 결과이며, 기본 구조에서는 learning rate가 성능에 큰 영향을 준다는 점을 확인할 수 있었다.

![SGD, BatchNorm False, Dropout False, lr=0.01, He](test_img/sgd_3.png)

### 3.4 SGD + BatchNorm + Dropout + He + lr=0.001

BatchNorm과 Dropout을 모두 적용했지만, 정확도는 `87.49%`로 오히려 낮아졌다.

파라미터 수는 `537,354`로 증가했는데, 이는 BatchNorm의 `gamma`, `beta` 파라미터가 추가되었기 때문이다.

loss curve도 다른 실험보다 높은 구간에서 천천히 감소했다. `lr=0.001`이 낮은 데다가 Dropout이 일부 뉴런을 끄기 때문에, SGD 환경에서는 학습이 더 느려진 것으로 보인다.

![SGD, BatchNorm True, Dropout True, lr=0.001, He](test_img/sgd_4.png)

### 3.5 SGD + BatchNorm + Dropout + Random(no scaling) + lr=0.001

He나 Xavier처럼 입력 차원에 맞춰 가중치 크기를 조절하는 초기화 스케일링을 제거하고, `np.random.randn(...)`으로 만든 랜덤값을 그대로 사용했다.

정확도는 `87.77%`로 He 초기화를 사용한 BatchNorm/Dropout 실험의 `87.49%`와 비슷했다.

다만 loss curve를 보면 loss가 높은 값에서 시작해 천천히 감소한다. 이는 가중치 초기값의 크기가 적절히 조절되지 않아 각 층을 지나는 신호의 분포가 불안정해지고, SGD가 파라미터를 조금씩만 수정하면서 느리게 학습한 결과로 볼 수 있다.

이 실험은 초기화 방법이 단순히 정확도만 바꾸는 것이 아니라, 학습 초반의 loss 크기와 수렴 속도에도 영향을 준다는 점을 보여준다.

![SGD, BatchNorm True, Dropout True, lr=0.001, Random no scaling](test_img/sgd_5.png)

## 4. 주요 비교

### 4.1 He와 Xavier 비교

| 초기화 | Accuracy |
| --- | --- |
| He | 90.70% |
| Xavier | 90.76% |

두 초기화 방식의 정확도 차이는 `0.06%p`로 매우 작았다.

ReLU에는 이론적으로 He 초기화가 더 적합하지만, 이번 실험에서는 모델이 깊지 않고 학습 조건이 단순해 차이가 크게 드러나지 않았다.

### 4.2 Learning Rate 비교

| Learning Rate | Accuracy |
| --- | --- |
| 0.001 | 90.70% |
| 0.01 | 95.77% |

SGD에서는 learning rate 변화가 가장 큰 영향을 주었다.

`lr=0.001`은 안정적이지만 학습 속도가 느렸고, `lr=0.01`은 더 빠르게 loss를 줄이며 정확도를 크게 높였다.

### 4.3 BatchNorm/Dropout 적용 결과

| BatchNorm | Dropout | Accuracy |
| --- | --- | --- |
| False | False | 90.70% |
| True | True | 87.49% |

BatchNorm과 Dropout을 추가했지만 성능이 바로 좋아지지는 않았다.

Dropout은 일부 뉴런을 끄는 규제 기법이므로 학습 속도를 늦출 수 있고, SGD와 낮은 learning rate 조합에서는 충분히 수렴하지 못할 수 있다.

따라서 BatchNorm/Dropout의 효과를 보려면 learning rate 조정, epoch 증가, Adam optimizer와의 조합 등을 함께 실험할 필요가 있다.

### 4.4 초기화 스케일링 제거 결과

| 초기화 | BatchNorm | Dropout | Accuracy |
| --- | --- | --- | --- |
| He | True | True | 87.49% |
| Random(no scaling) | True | True | 87.77% |

초기화 스케일링을 제거했을 때 최종 정확도는 크게 무너지지 않았지만, loss가 높은 구간에서 천천히 감소했다.

이는 BatchNorm이 어느 정도 신호 분포를 보정해 주기 때문에 정확도 차이는 작게 나타났지만, 적절한 가중치 초기화가 없으면 학습 초반의 안정성과 수렴 속도에는 영향을 줄 수 있음을 보여준다.

## 5. 결론

이번 SGD 실험에서 가장 성능이 좋았던 조합은 `SGD + He + lr=0.01 + BatchNorm False + Dropout False`였다.

단순히 정규화 기법을 추가한다고 항상 정확도가 오르는 것은 아니며, optimizer와 learning rate가 함께 맞아야 효과가 나타난다는 점을 확인했다.

특히 SGD에서는 learning rate가 성능에 큰 영향을 주었고, `0.001`보다 `0.01`에서 훨씬 빠르게 loss가 감소하며 목표 정확도인 95%를 넘겼다.

또한 초기화 스케일링을 제거한 실험에서는 최종 정확도 차이는 크지 않았지만, loss가 천천히 감소하는 모습을 확인했다. 이를 통해 가중치 초기화는 최종 정확도뿐 아니라 학습 속도와 안정성에도 영향을 준다는 점을 확인할 수 있었다.

## 6. 레포트용 요약 문장

SGD 기반 실험에서는 learning rate가 가장 큰 영향을 주었다. `lr=0.001`에서는 He와 Xavier 초기화 모두 약 90% 수준의 정확도를 보였고, 초기화 방식에 따른 차이는 크지 않았다. 반면 `lr=0.01`로 조정했을 때 정확도는 95.77%까지 상승했다. BatchNorm과 Dropout을 함께 적용한 경우에는 파라미터 수가 증가했지만, 낮은 learning rate와 Dropout의 규제 효과로 인해 오히려 학습이 느려지고 정확도가 87%대에 머물렀다. 초기화 스케일링을 제거한 실험에서는 최종 정확도는 크게 달라지지 않았지만 loss가 높은 값에서 천천히 감소해, 적절한 초기화가 학습 속도와 안정성에 영향을 준다는 점을 확인했다. 이를 통해 모델 성능은 특정 기법 하나가 아니라 learning rate, optimizer, 초기화 방식, 정규화 기법의 조합에 따라 달라진다는 점을 확인했다.

---

# Adam 실험 결과 정리

## 1. 실험 목적

이번 실험은 optimizer를 `Adam`으로 고정하고, BatchNorm/Dropout 적용 여부와 Dropout 비율에 따라 학습 결과가 어떻게 달라지는지 확인하는 것을 목표로 한다.

활성화 함수는 모든 실험에서 `ReLU`를 사용했고, 출력층은 `Softmax`를 사용했다.

## 2. 실험 조건

| 번호 | Optimizer | BatchNorm | Dropout | Dropout Ratio | 초기화 | Test Accuracy | Total Params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Adam | False | False | 0.5 | He | 98.08% | 535,818 |
| 2 | Adam | True | True | 0.5 | He | 98.42% | 537,354 |
| 3 | Adam | True | True | 0.3 | He | 98.35% | 537,354 |

## 3. 실험별 관찰

### 3.1 Adam + He + BatchNorm False + Dropout False

Adam optimizer만 적용하고 BatchNorm과 Dropout은 사용하지 않았다.

정확도는 `98.08%`로, SGD 실험의 최고 정확도인 `95.77%`보다 높았다.

loss curve도 초반에 매우 빠르게 감소했고, 이후 낮은 loss 구간에서 작은 변동을 보였다. 이는 Adam이 gradient의 이동평균과 제곱 이동평균을 사용해 SGD보다 빠르고 안정적으로 파라미터를 갱신했기 때문으로 해석할 수 있다.

![Adam, BatchNorm False, Dropout False, He](test_img/adam_1.png)

### 3.2 Adam + BatchNorm True + Dropout True + ratio=0.5 + He

BatchNorm과 Dropout을 모두 적용했고, Dropout 비율은 `0.5`로 설정했다.

정확도는 `98.42%`로 Adam 실험 중 가장 높았다.

파라미터 수는 `537,354`로 증가했는데, 이는 BatchNorm의 `gamma`, `beta` 파라미터가 추가되었기 때문이다.

loss는 초반에 빠르게 감소한 뒤 안정적으로 내려갔다. Dropout 비율이 높아 학습 loss 자체는 BatchNorm/Dropout이 없는 경우보다 높게 유지되지만, test accuracy는 더 좋아졌다.

![Adam, BatchNorm True, Dropout True, ratio=0.5, He](test_img/adam_2.png)

### 3.3 Adam + BatchNorm True + Dropout True + ratio=0.3 + He

Dropout 비율을 `0.5`에서 `0.3`으로 낮췄다.

정확도는 `98.35%`로 `ratio=0.5`의 `98.42%`와 큰 차이는 없었다.

loss curve는 더 낮은 loss까지 내려갔지만, 최종 test accuracy는 거의 비슷했다. 이는 Dropout 비율 변화가 현재 모델과 MNIST 데이터셋에서는 큰 성능 차이로 이어지지 않았다는 것을 보여준다.

![Adam, BatchNorm True, Dropout True, ratio=0.3, He](test_img/adam_3.png)

## 4. 주요 비교

### 4.1 Adam과 SGD 최고 결과 비교

| Optimizer | 주요 조건 | Best Accuracy |
| --- | --- | --- |
| SGD | BatchNorm False, Dropout False, lr=0.01, He | 95.77% |
| Adam | BatchNorm True, Dropout True, ratio=0.5, He | 98.42% |

Adam은 SGD보다 더 높은 정확도에 도달했다.

특히 Adam은 learning rate를 크게 조정하지 않아도 loss가 빠르게 감소했고, 98% 이상의 정확도에 도달했다.

### 4.2 BatchNorm/Dropout 적용 여부 비교

| 조건 | Accuracy |
| --- | --- |
| Adam + BatchNorm False + Dropout False | 98.08% |
| Adam + BatchNorm True + Dropout True | 98.42% |

Adam에서는 BatchNorm과 Dropout을 추가했을 때 정확도가 소폭 상승했다.

SGD에서는 BatchNorm/Dropout을 켰을 때 오히려 성능이 떨어졌지만, Adam과 함께 사용할 경우 더 안정적으로 학습되어 성능 향상으로 이어졌다.

### 4.3 Dropout Ratio 비교

| Dropout Ratio | Accuracy |
| --- | --- |
| 0.5 | 98.42% |
| 0.3 | 98.35% |

Dropout 비율을 `0.5`에서 `0.3`으로 바꿨을 때 정확도 차이는 `0.07%p`로 매우 작았다.

따라서 현재 모델과 MNIST 데이터셋에서는 Dropout 비율 변화가 큰 성능 차이를 만들지는 않았다.

## 5. 결론

Adam 실험에서는 모든 조합이 `98%` 이상의 높은 정확도를 보였다.

가장 좋은 결과는 `Adam + BatchNorm True + Dropout True + ratio=0.5 + He` 조합의 `98.42%`였다.

SGD 실험과 비교하면 Adam은 훨씬 빠르게 loss를 줄였고, 최종 정확도도 높았다.

BatchNorm과 Dropout은 Adam과 함께 사용할 때 성능을 소폭 개선했지만, Dropout 비율 변화는 큰 차이를 만들지 않았다.

## 6. 레포트용 요약 문장

Adam 기반 실험에서는 모든 조합이 98% 이상의 정확도를 기록했다. BatchNorm과 Dropout을 사용하지 않은 기본 Adam 모델도 98.08%의 높은 정확도를 보였고, BatchNorm과 Dropout을 함께 적용했을 때 98.42%로 가장 좋은 결과를 얻었다. Dropout 비율을 0.5에서 0.3으로 낮췄을 때 정확도 차이는 거의 없었다. 이를 통해 Adam optimizer가 SGD보다 빠르고 안정적으로 수렴하며, BatchNorm과 Dropout은 Adam과 함께 사용할 때 성능을 소폭 개선할 수 있음을 확인했다.

## 7. 전체 실험 종합 결론

SGD 실험에서는 learning rate가 가장 큰 영향을 주었고, `lr=0.001`에서는 90% 수준에 머물렀지만 `lr=0.01`에서는 95.77%까지 상승했다.

Adam 실험에서는 모든 조합이 98% 이상의 정확도를 보였고, optimizer 변경만으로도 큰 성능 향상이 나타났다.

SGD 실험에서 He와 Xavier 초기화 차이는 크지 않았다.

초기화 스케일링을 제거한 SGD 실험에서는 loss 감소가 느려져, 가중치 초기화가 학습 속도와 안정성에 영향을 준다는 점을 확인했다.

Dropout 비율 변화 역시 현재 모델과 MNIST 데이터셋에서는 큰 차이를 만들지 않았다.

최종적으로 가장 성능이 좋았던 조합은 `Adam + BatchNorm True + Dropout True + ratio=0.5 + He`였으며, test accuracy는 `98.42%`였다.
