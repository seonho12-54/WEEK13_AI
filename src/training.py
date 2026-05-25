# -*- coding: utf-8 -*-
"""학습 루프, 평가, 시각화 함수 모음."""

import matplotlib.pyplot as plt
import numpy as np

from losses import cross_entropy_loss


def train(model, optimizer, x_train, y_train, epochs=20, batch_size=128):
    """
    미니배치 학습 루프.

    한 배치마다 Forward -> Loss -> Backward -> Optimizer 업데이트 순서로 진행합니다.
    교육생은 이 함수에서 "예측값을 만들고, 손실을 계산하고, gradient로 파라미터를 바꾸는"
    전체 흐름을 확인할 수 있습니다.

    Returns:
        loss_history: epoch별 평균 손실 리스트
    """
    # TODO: epoch마다 데이터를 섞고, batch 단위로 forward/loss/backward/update를 수행하세요.
    # 힌트: Softmax + CrossEntropy 결합 gradient는 y_pred copy에서 정답 위치에 1을 빼서 만듭니다.
    #raise NotImplementedError("train을 구현하세요.")

    #epoch마다 평균 loss를 저장할 리스트
    loss_history = []
    #학습 데이터가 총 몇개인지
    train_size = x_train.shape[0]
    #설정한 epoch만큼 전체 데이터를 반복하여 학습
    for epoch in range(epochs):
        #학습 데이터를 매 epoch마다 셔플
        indices = np.random.permutation(train_size)
        #이번 epoch의 loss합계를 저장할 변수
        epoch_loss = 0.0
        #이번 epoch에서 batch를 몇 번 처리했는지 저장할 변수
        batch_count = 0
        #전체 데이터를 batch_size만큼 잘라서 반복
        for start in range(0, train_size, batch_size):
            end = start + batch_size
            #섞인 index중에 현재 batch에 해당하는 index만 뽑음
            batch_indices = indices[start:end]
            #현재 batch의 입력데이터와 정답 라벨을 꺼냄
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]
            #모델에 입력을 넣어서 예측확률을 구함, 학습 중엔 True
            y_pred = model.forward(x_batch, train = True)
            #교차앤트로피로 오차 계산
            loss = cross_entropy_loss(y_pred, y_batch)
            #현재 batch에 실제 데이터가 몇개인지(마지막 batch는 128개보다 적을 수 있으므로)
            current_batch_size = x_batch.shape[0]
            #softmax + cross_entropy_error의 gradient를 만들기 위해 예측값 복사
            dout = y_pred.copy()
            #각 샘플의 정답 클래스 위치에서 1을 뺀다. 왜?
            dout[np.arange(current_batch_size), y_batch] -= 1
            #batch_size로 나누어서 평균 gradient를 계산
            dout /= current_batch_size
            #역전파 실행, 각 파라미터의 gradient계산
            model.backward(dout)
            #각 파라미터의 gradient를 참고하여 업데이트
            optimizer.update(model.params, model.grads)
            #이번 batch의 loss를 loss합계에 더함
            epoch_loss += loss
            #처리한 배치 수 1증가
            batch_count += 1
        #이번 epoch의 평균 loss를 저장
        loss_history.append(epoch_loss / batch_count)
    #epoch별 평균 loss 목록을 반환, loss그래프에 사용
    return loss_history


def evaluate(model, x, y):
    """정확도(%)와 총 파라미터 수 반환."""
    y_pred = model.predict(x)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(p.size for p in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history):
    """손실 커브 그래프."""
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
