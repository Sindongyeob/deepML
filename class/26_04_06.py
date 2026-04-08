#4월 6일 수업 내용 정리

#CNN: convolutional neural network
#신경망에서 상위 레이어와 하위 레이어가 부분적으로 연결되어있는 신경망
#이미지 처리에 특히 적합함, 대표적인 CNN이 AlexNet

#CNN 구조
#입력층에서 convolution 연산을 통해 feature map을 생성
#pooling 연산을 통해 feature map의 크기를 줄임, pooling은 입력차원을 줄이는 역할
#convolution과 pooling을 반복후 flattening을 통해 1차원 벡터로 변환 후에
#맨 끝에는 fully connected layer가 위치하여 최종적으로 분류 작업을 수행

#convolution 연산: '주변 화소값들에 가중치를 곱하여 더한 후' 출력값을 계산하는 연산
#필터 = 커널 = 마스크
#특징: 부분 연결, 겹치는 영역 조정(stride), 동일한 커널 사용(동일 가중치) -> 가중치 수 감소

#stride: 커널 적용 시 이동하는 간격(1이면 한 픽셀씩 이동)
#padding(subsampling): 입력 이미지 가장자리 처리 기법
# -valid: padding 없음, 출력 크기 '감소'
# -same: padding 있음 가장자리를 특정값으로 채움(0 or 이웃값), 출력 크기 '유지'

#종류
#max pooling: 영역 내 최대값 선택
#average pooling: 영역 내 평균값 선택

#장점
#물체의 이동에 둔감하다 ex) 물체가 1픽셀만큼 움직여도 풀링 연산값은 동일

#convolution dimension 계산 공식
#출력 크기 = (input_size - kernel_size + 2*padding거리) / stride + 1

#필터가 여러개 일때 컨볼루션 레이어
#***필터 값은 학습이 되는 것***
#필터가 n개면 출력인 feature map도 n개가 됨
#feature map이 여러개인 경우 box형태로 표현

#convolution 함수
import tensorflow as tf
from tensorflow.keras import layers
#tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), activation=None, input_shape, padding='valid’)
#filters: 필터의 수, kernel_size: 필터의 크기, strides: 이동 간격, activation: 활성화 함수, input_shape: 입력 데이터 형태, padding: 패딩 방식(valid, same)

layers.Conv2D(10,kernel_size=(3,3),activation='relu',padding='same',input_shape=(28,28,1))
#--> 필터 10개, 커널 크기 3x3, 활성화 함수 ReLU, 패딩 방식 same, 입력 데이터 형태 28x28x1(흑백 이미지)
#출력 feature map의 크기는 (28, 28, 10) -> 28x28 크기의 feature map이 10개 생성됨
#padding이 same이므로 입력과 출력의 공간적 크기가 동일하게 유지됨, valid이면 출력 크기가 (26, 26, 10) -> 커널이 3x3이므로 가장자리 1픽셀씩 제거되어 출력 크기가 감소됨
#padding default는 valid