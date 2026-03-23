#26년 3월 23일 수업 정리
#가중치 업데이트 방법 
#SGD(Stochastic Gradient Descent): 전체 데이터셋에서 무작위로 선택된 하나의 샘플에 대해 가중치 업데이트, 빠르지만 노이즈가 많음
#full batch gradient descent: 전체 데이터셋을 사용하여 가중치 평균 업데이트, 안정적이지만 느림, local minimum 탈출 어려움
#mini-batch gradient descent: 전체 데이터셋을 작은 배치로 나누어 각 배치에 대해 가중치 업데이트, SGD와 full batch의 장점 결합

#learning rate: 가중치를 얼마나 변경할 것인가, 모델 성능에 큰 영향을 미침, 설정하기 어려움

#momemtum: 가중치 업데이트에 관성을 추가하여 최적화 과정에서 진동을 줄이고 수렴 속도를 높이는 방법, 이전 업데이트의 영향을 고려하여 가중치 업데이트

#adagrad(apadaptive gradient): 학습률을 각 매개변수에 대해 적응적으로 조정하는 방법
#자주 학습된 방향은 덜 움직이고, 덜 학습된 방향은 더 많이 움직임 -> 학습률 감쇠(learning rate decay)
#학습률 설정: 이전단계의 기울기들을 누적한 값에 반비례하여 설정

#RMSprop(root mean square propagation): adagrad의 단점을 보완한 방법, 
#gradient 누적 대신 EMA(exponential moving average; 지수 가중 이동 평균)를 사용하여 학습률 조정
#최근 데이터에는 더큰 가중치를 주고, 과거 데이터에는 지수적으로 감소하는 가중치를 주는 평균 이동 방법
#학습률이 너무 빨리 감소하는 문제 해결

#Adam(adaptive moment estimation): momentum과 RMSprop을 결합한 방법, 적응적 학습률과 모멘텀을 모두 활용하여 가중치 업데이트
#현대 딥러닝에서 가장 널리 사용되는 optimizer

import numpy as np 
#tensorflow: 딥러닝 프레임워크, 딥러닝에서 배열을 tensor로 표현
import tensorflow as tf
#keras: tensorflow 위에서 작동하는 고수준 딥러닝 API, 모델 구축과 훈련을 간단하게 만들어줌
#핵심 데이터 구조는 model이며 layer를 구성하는 방법을 나타냄
#가장 간단한 모델은 Sequential 모델, layer를 순차적으로 쌓는 방식

X=np.array([[0,0],[0,1],[1,0],[1,1]]) #입력 데이터
y=np.array([[0],[1],[1],[0]]) #XOR 문제의 정답

#sequential 모델 생성
model = tf.keras.models.Sequential()

#layer 추가, units는 노드 개수
model.add(tf.keras.layers.Dense(units=2, input_shape=(2,), activation='sigmoid')) #은닉층, 2개의 유닛, 입력 크기 2, 활성화 함수 sigmoid
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #출력층, 1개의 유닛, 활성화 함수 sigmoid

#모델 손실함수및 최적화 정의
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(learning_rate=0.3)) #손실 함수 MSE, 옵티마이저 SGD

model.fit(X, y, epochs=1000) #모델 훈련, 입력 X, 정답 y, 1000번 반복

print(model.predict(X)) #모델 예측
#model.summary() 모델 요약본 보여줌

#케라스 사용방법 3가지
# 1. Sequential 모델을 만들고 모델에 필요한 레이어를 추가하는 방법
# model = Sequential()
# model.add(Dense(units=2, input_shape=(2,), activation='sigmoid')) 
# model.add(Dense(units=1,  activation='sigmoid'))

# 2. 함수형 API를 사용하는 방법
# inputs = Input(shape=(2,))
# x = Dense(2, activation="sigmoid")(inputs)
# # 입력층
# # 은닉층 ①
# prediction = Dense(1, activation="sigmoid")(x) # 출력층
# model = Model(inputs=inputs, outputs=prediction)

# 3. model 클래스 상속받아서 클래스 정의

#hyper parameter: 학습률이나 은닉층은 몇개로 할 것인가
#학습률, 모멘텀 가중치, 은닉층 개수, 유닛의 개수, 미니 배치 크기

#방법
#기본값 사용: 라이브러리 개발자가 설정한 default값 사용
#수동 검색: 직접 지정
#그리드 검색: 하이퍼 파라미터 변경하면서 성능 측정
#random




