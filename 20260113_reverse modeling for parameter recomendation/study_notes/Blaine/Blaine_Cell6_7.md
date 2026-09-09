# Cell 6
Parametric Study 주석


# Cell 7
"def build_model(input_shape):"
build_model이라는 함수를 정의하고, input_shape을 입력으로 사용 (build_model - 모델 구조 정의)

"model = Sequential"
순서대로 층을 쌓는 모델 생성

"Dense(128, activation='swish', kernel_intializer='glorot_uniform', input_shape=input_shape,
BatchNormalization(),
Dropout(0.1), ....)"
 - Dense(128): 노드 128개짜리 층
 - activation='swish': 각 노드의 출력에 swish 함수 적용
 - kernel_initializer='glorot_uniform': 가중치 초기값을 glorot 방식으로 설정. 학습이 안정적으로 시작되도록 처음 가중치를 너무 크거나 직지 않게 설정하는 방법. 앞 층과 뒤 층의 노드 수를 보고 자동으로 적절한 범위 계산
 - input_shape=input_shape: 첫 번째 층이라 입력 크기를 명시

 "Dense(1)"
 마지막 출력층. 최종적으로 Blaine 값 하나만 예측하기 때문에 노드 1개. 범위 제한 없이 숫자 그대로 출력하기 때문에 활성화함수 없음

 "input_shape = (38, )
 model = build_model(input_shape)"
 (38, )를 build_model에 넣어서 호출하고, 반환된 모델을 model에 저장 (keras가 입력 크기를 튜플로 받기 때문에 몇 차원인지 알려줘야 함)

 "huber_loss = tf.keras.loss.Huber(delta=1.0)"
 학습할 때 예측값과 실제값의 차이를 계산하는 방식. Huber loss는 오차가 작으면 MSE(오차 제곱)처럼, 오차가 크면 MAE(평균 절대 오차)처럼 동작해 이상치에 덜 민감. delta=1.0은 오차가 1보다 작으면 MSE, 크면 MAE 방식으로 전환하라는 기준값

 "model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=huber_loss)"
 모델 학습 준비를 마치는 두 단계 (model.compile - 학습 방법 정의)
 - optimizer=Adam(learning_rate=0.0005): 가중치를 어떻게 업데이트할지 방법을 지정. Adam은 학습 속도를 자동으로 조절해주는 방식
 - loss=huber_loss: 앞에서 만든 Huber Loss를 오차 계산 방식으로 이용

 "checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model_blaine.h5',
    monitor = 'val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
 )"
 - 'best_model_blaine.h5': 저장할 파일명
 - monitor='val_loss': 검증 손실을 기준으로 판단
 - save_best_only=True: 가장 낮은 val_loss 모델만 저장
 - save_weights_only=False: 가중치만이 아닌 모델 전체 저장
 - mode='min': val_loss가 작을수록 좋다는 기준
 - verbose=1: 저장될 때마다 터미널 출력

 "early_stopping = EarlyStopping(monitor='val_loss', patience=300, verbose=1)"
 학습을 자동으로 멈추는 설정. 
 - monitor='val_loss': 검증 손실을 기준으로 판단
 - patience=300: val_loss가 300 에포크 동안 개선되지 않으면 학습 중단
 - verbose=1: 중단될 때 터미널 출력

 "model.summary()"
 모델 구조를 터미널에 출력. (층 이름, 노드 수, 학습 가능한 가중치 수 등)
