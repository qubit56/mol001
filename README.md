# mol001
# 크리스탈의 비밀

제 1 장 마법의 크리스탈 볼과 딥러닝

[STORY]

상상해보세요, 우리에게는 특별한 "마법의 크리스탈 볼"이 있다고 해요. 이 볼은 미래의 일을 예측할 수 있어요. 우리는 이 볼을 이용해서 미래의 날씨와 생수 판매량을 예측하려고 해요.

하지만 이 크리스탈 볼은 바로 사용할 수 있는 것이 아니에요. 우리는 크리스탈 볼에게 '학습'이라는 과정을 거쳐서 볼을 사용법을 가르쳐줘야 해요. 이때 필요한 것이 바로 '딥러닝'입니다.

딥러닝으로 크리스탈 볼 교육하기

데이터 수집: 먼저, 크리스탈 볼에게 가르쳐줄 과거의 날씨 데이터와 그 때의 생수 판매량 데이터를 수집해요.
볼에게 데이터 보여주기: 수집한 데이터를 크리스탈 볼에게 보여줍니다. "이 날은 더웠어. 그래서 생수가 많이 팔렸어" 혹은 "이 날은 춥고 비가 왔어. 그래서 생수 판매량이 적었어"라고 알려줘요.
예측 연습: 크리스탈 볼에게 어느 날의 날씨 정보만 주고, 그 때의 생수 판매량을 어떻게 될지 예측하게 해보아요.
교정하기: 크리스탈 볼의 예측이 틀리면 바로잡아서, 더 나은 예측을 할 수 있도록 교육시켜요.
이렇게 반복적으로 교육을 진행하면, 크리스탈 볼(딥러닝 모델)은 미래의 날씨를 보고 생수 판매량을 어느 정도 예측할 수 있게 돼요.

[실습 코드]

파이썬의 TensorFlow와 Keras 라이브러리를 사용해서 간단한 딥러닝 모델을 만들어 예측하는 과정

import numpy as np
import tensorflow as tf
from tensorflow import keras

# np.random.seed를 사용하여 재현 가능한 랜덤 데이터 생성
np.random.seed(42)

# 가상의 날씨 데이터 생성:
# 온도는 20~35도 사이, 습도는 50~90% 사이의 랜덤 값으로 30개 생성
weather_data = np.column_stack([
    np.random.randint(20, 35, 30),  # 온도 데이터
    np.random.randint(50, 90, 30)   # 습도 데이터
])

# 가상의 생수 판매량 데이터 생성:
# 실제 판매량과는 관련이 없지만, 단순히 예제로 사용하기 위한 랜덤 데이터 생성
# 50~150 사이의 랜덤 값으로 30개 생성
water_sales = np.random.randint(50, 150, 30)
print("생수 판매량 : ", water_sales)

# 딥러닝 모델 구조 정의
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(2,)),  
# 입력 레이어: 2개의 특성(온도, 습도) 받음
    keras.layers.Dense(64, activation='relu'),                    
# 은닉 레이어: 64개의 노드로 구성
    keras.layers.Dense(1)                                        
# 출력 레이어: 생수 판매량 1개의 값 출력
])

# 모델 컴파일: 최적화 알고리즘과 손실 함수 설정
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습: weather_data를 입력으로, water_sales를 출력으로 하는 데이터로 100번의 에포크로 학습
model.fit(weather_data, water_sales, epochs=10)

# 예측하기: 미래의 어느 날의 날씨 데이터(예: [32, 65])로 생수 판매량 예측
future_weather = np.array([[32, 65]])
predicted_sales = model.predict(future_weather)
print(f"예측된 생수 판매량: {predicted_sales[0][0]}")


[코드 해설]

1. import numpy as np
**Numpy (Numerical Python)**는 파이썬에서 수치 계산을 위한 핵심 라이브러리입니다.

용도:

다차원 배열 객체와 배열과 관련된 다양한 연산 기능을 제공합니다.
행렬 연산, 통계, Fourier 변환, 선형대수 등 수학 연산을 지원합니다.
주요 기능:

np.array(): 리스트를 다차원 배열로 변환합니다.
np.zeros(), np.ones(): 모든 요소가 0 또는 1인 배열을 생성합니다.
np.random.rand(): 난수를 가진 배열을 생성합니다.
np.dot(): 행렬 곱을 계산합니다.

2. import tensorflow as tf
TensorFlow는 머신 러닝과 딥 러닝을 위한 오픈 소스 라이브러리입니다.

용도:

데이터 플로우 그래프를 사용하여 수치 연산을 표현합니다.
CPU와 GPU에서 실행 가능합니다.
주요 기능:

tf.constant(), tf.Variable(): TensorFlow에서 사용되는 상수와 변수를 정의합니다.
tf.GradientTape(): 자동 미분(그래디언트 계산)을 위한 API입니다.
tf.data.Dataset: 대량의 데이터를 효율적으로 처리하기 위한 데이터 파이프라인을 생성합니다.

3. from tensorflow import keras
Keras는 딥 러닝 모델을 쉽게 만들고 학습시키기 위한 고수준 인터페이스입니다. 
TensorFlow 2.0부터는 TensorFlow의 공식 API로 포함되어 있습니다.

용도:

신경망 모델을 빠르게 설계, 훈련, 평가 및 예측을 수행합니다.
주요 기능:

keras.Sequential(): 층을 순차적으로 쌓아 딥러닝 모델을 구성합니다.
keras.layers.Dense(), keras.layers.Conv2D(): 다양한 유형의 뉴럴 네트워크 레이어를 제공합니다.
keras.optimizers, keras.losses: 최적화 알고리즘과 손실 함수들을 제공합니다.

4. Relu

ReLU (Rectified Linear Unit) 함수의 수식은 간단합니다. 주어진 입력 
x에 대해, 함수는 다음과 같이 정의됩니다:

간단히 말하면, 입력 값이 0보다 크면 그 값을 그대로 반환하고, 0 이하이면 0을 반환합니다.

[라이브러리 변환]

https://www.tensorflow.org/js/tutorials/conversion/import_keras?hl=ko




