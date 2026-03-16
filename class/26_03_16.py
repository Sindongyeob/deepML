#26년 3월 16일 수업 정리
#perceptron의 한계: 선형 분리 문제만 해결 가능(XOR 문제 해결 불가능)
#--> MLP, input layer와 output layer 사이에 hidden layer 추가하여 해결 가능

#MLP(Multi-Layer Perceptron): 여러 층의 퍼셉트론으로 구성된 신경망 모델
#입력층(input layer), 은닉층(hidden layer), 출력층(output layer)으로 구성

#activation function: 입력의 총합을 받아 출력값을 계산(출력 노드의 활성화 여부 결정)
#활성화 함수의 역할: 선형 결과를 비선형 변환으로 바꿔주는 역할

#perceptron에서는 step function 사용, mlp에서는 사용하지 않음
#단점 - step function은 미분 불가능, backpropagation 알고리즘에서 가중치 업데이트 어려움, 출력이 단순하여 연속적인 확률 값 표현 불가능

#mlp 활성화 함수: sigmoid, tanh, ReLU(가장 많이 사용) 등 사용
#sigmoid: 출력값이 0과 1 사이, 확률로 해석 가능, vanishing gradient 문제 발생 가능
#tanh: 출력값이 -1과 1 사이, 출력값이 음수가 필요할 때 사용, vanishing gradient 문제 발생 가능
#ReLU(Rectified Linear Unit): 출력값이 0과 양수, 음수 입력에 대해 0 출력, 계산 비용 매우 낮음, gradient 유지됨

#예제 활성화 함수 구현
import numpy as np
import matplotlib.pyplot as plt

#step function
def step_function(t):
    return np.where(t > 0, 1, 0) #t가 0보다 크면 1, 그렇지 않으면 0 반환

#sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#tanh function
def tanh(x):
    return np.tanh(x)
#ReLU function
def relu(x):
    return np.maximum(0, x) #0보다 작으면 0,0보다 크면 x 그대로 반환

#활성화 함수 그래프 그리기
x = np.linspace(-10, 10, 100) #-10에서 10까지 100개의 점 생성
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.plot(x, step_function(x))
plt.title("Step Function")

plt.subplot(1, 4, 2)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid Function")   

plt.subplot(1, 4, 3)
plt.plot(x, tanh(x))    
plt.title("Tanh Function")

plt.subplot(1, 4, 4)
plt.plot(x, relu(x))
plt.title("ReLU Function")
plt.show()

#mlp 순방향 패스(foward propagation): 입력이 각 층을 거쳐 출력으로 전달되는 과정, 입력층 -> 은닉층 -> 출력층
#순방향 패스 구현
def forward_pass(X, W1, W2, b):
    #입력층에서 은닉층으로
    z1 = np.dot(X, W1)+[b[0],b[1]] #입력과 가중치의 내적 + bias
    a1 = sigmoid(z1) #은닉층 활성화 함수 적용

    #은닉층에서 출력층으로
    z2 = np.dot(a1, W2)+[b[2]] #은닉층 출력과 가중치의 내적 + bias
    a2 = sigmoid(z2) #출력층 활성화 함수 적용
    
    return a2 #최종 출력 반환

X = np.array([[0,0],[0,1],[1,0],[1,1]]) #입력 데이터
W1 = np.random.rand(2, 2) #입력층에서 은닉층으로의 가중치
W2 = np.random.rand(2, 1) #은닉층에서 출력층
b = np.random.rand(3) #bias
y = np.array([[0],[1],[1],[0]]) #XOR 문제의 정답
output = forward_pass(X, W1, W2, b)
print(output)# 오차계산 및 업데이트가 없어서 단순 난수가 출력됨

#손실 함수(loss function): 모델의 예측값과 실제값의 차이를 측정하는 함수, 학습의 성과를 나타내는 지표
#MSE: 평균 제곱 오차
def mse(pred, y):
    return np.mean((pred - y) ** 2)/len(pred)

#역전파 알고리즘
#grdient descent: 경사 하강법, mlp 가중치 업데이트 학습 알고리즘
#가중치 업데이트 공식: W(t+1) = W(t) - learning_rate * gradient
#gradient: 손실 함수의 가중치에 대한 편미분

#역전파 알고리즘에서 gradient descent는 출력층에서 손실함수를 한 번만 계산한 후 이것을 역전파하여 모든 위치에서 가중치 업데이트에 사용한다(chain rule)


