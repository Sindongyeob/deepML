#26년 3월 9일 수업 정리

#기본 개념 정리
#인공지능: 컴퓨터가 인간처럼 행동하도록 만드는 기술
#머신러닝: 데이터를 이용하여 함수를 학습하는 기술
#딥러닝: 심층 신경망을 사용하는 머신러닝 기술

#전통적인 알고리즘은 출력이 데이터이고 사람이 규칙을 만들지만
#머신러닝은 입력 데이터와 출력 데이터로 컴퓨터가 규칙을 만든다 --> 입력을 받아 함수를 학습(함수 근사)

#머신러닝 종류
#지도학습: 입력과 정답(label)이 있는 데이터로 학습: regression, classification
#비지도학습: 입력만 있는 데이터로 학습: clustering, dimensionality reductio
#강화학습: 행동과 보상으로 학습: 게임, 로봇 제어

#5장 perceptron
#perceptron: 인공 신경망
#여러 입력 정보를 이용해 두가지 클래스(1 or 0) 중 하나를 선택하는 가장 단순한 분류 모델
#bias: 값들이 한 쪽으로 치우치는 것을 방지하기 위해 추가하는 값

#perceptron 예제
import numpy as np

epsilon = 1e-7#부동소수점 오차 방지

def step_function(t): #퍼셉트론 활성화 함수
    if t > epsilon: return 1
    else: return 0

#입력 데이터, 각 리스트의 맨 끝의 1은 bias 항을 나타냄
X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])    

#정답 레이블
y = np.array([0,0,0,1]) #AND 
W = np.random.rand(len(X[0])) #가중치 초기화
np.set_printoptions(precision=3)#소수점 셋째자리 까지의 난수

#퍼셉트론 학습 알고리즘
def perceptron_fit(X,Y, epochs=10):
    global W #가중치 업데이트를 위해 전역변수로 선언
    eta = 0.2 #학습률

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        for i in range(len(X)):
            predict = step_function(np.dot(X[i], W)) #예측값 계산, np.dot(X[i], W) 는 입력과 가중치의 내적
            error = Y[i] - predict #오차 계산
            W += eta * error * X[i] #가중치 업데이트
            print("Weights:", W)

#예측
def perceptron_predict(X, Y):
    global W
    for x in X:
        print(x[0], x[1], "->", step_function(np.dot(x, W)))

#실행
perceptron_fit(X, y, epochs=10)
perceptron_predict(X, y)


#scikit-learn 라이브러리를 이용한 퍼셉트론 구현
from sklearn.linear_model import Perceptron
X = [[0,0],
     [0,1],
     [1,0],
     [1,1]]

y = [0,0,0,1] #AND
print("scikit-learn Perceptron")
clf = Perceptron(tol=1e-3, random_state=0) #tol: 수렴 기준, random_state: 난수 시드
clf.fit(X, y) #모델 학습
print(clf.predict(X)) #예측 결과 출력
print("Weights:", clf.coef_) #가중치 출력


#퍼셉트론의 한계: 선형 분리 문제
#XOR 문제: 입력이 (0,0), (0,1), (1,0), (1,1) 일 때 출력이 0, 1, 1, 0 인 문제
#퍼셉트론은 선형적으로 분리 가능한 문제만 해결할 수 있기 때문에 XOR 문제는 해결할 수 없다
#--> 다층 퍼셉트론(Multi-layer Perceptron, MLP)으로 해결 가능