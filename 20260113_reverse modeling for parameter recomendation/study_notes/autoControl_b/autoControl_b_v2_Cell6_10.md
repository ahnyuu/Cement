# Cell 6 - 데이터를 0~1 사이로 정규화하는 도구(scaler)를 만들고 저장하는 셀

scaler_x: 품질 모델 입력 38개용
scaler_y: Blaine 값(출력)용

"scaler_x = MinMaxScaler(feature_range=(0.0, 0.1))
scaler_x.fit(df.loc[idx_all, QUALITY_INPUT_COLS].to_numpy(dtype=np.floate32))"
38개 입력 변수를 0~1로 정규화할 도구를 만들고, 전체 데이터에서 각 변수의 최솟값/최댓값을 학습
- MinMaxScaler(feature_range=(0.0, 1.0)): 최솟값은 0, 최댓값은 1로 변환하는 도구 생성 (scaler_x)
- df.loc[idx_all, QUALITY_INPUT_COLS]:  df에서 38개 열만 꺼냄. (df.loc[행,열])
- .to_numpy(dtype=np.float32): 표 형태를 숫자 배열로 변환
- .fit: 각 열의 최솟값/최댓값 학습

"blaine_idx = QUALITY_INPUT_COLS.index("blaine")
CUSTOM_MIN = 3030
CUSTOM_MAX = 5500

for idx in [blaine_idx]:
    scaler_x.data_min_[idx] = CUSTOM_MIN
    scaler_x.data_max_[idx] = CUSTOM_MAX
    scaler_x.data_rnage_[idx] = CUSTOM_MAX - CUSTOM_MIN
    scaler_x.scale_[idx] = 1.0 / (CUSTOM_MAX - CUSTOM_MIN)
    scaler_x.min_[idx] = -CUSTOM_MIN / (CUSTOM_MAX - CUSTOM_MIN)"
scaler_x가 자동으로 학습한 blaine의 최솟값/최댓값을 수동으로 덮어써서 정규화 범위를 강제로 지정

"joblib.dump(scaler_x, SCALER_X_PATH)
joblib.dump(scaler_y, SCALER_Y_PATH)"
만든 두 스케일러를 파일로 저장
- joblib: 파이썬 객체를 파일로 저장/불러오는 라이브러리
- joblib.dump: 객체를 파일로 저장하는 함수. dump(저장할 것, 저장경로)


# Cell 7
"quality_model = tf.keras.models.load_model(QUALITY_MODEL_PATH)
quality_model.trainable = False

print("Quality model input/output:", quality_model.input_shape, quality_model.output_shape)"
미리 학습된 품질 예측 모델을 불러오고, 더 이상 학습되지 않도록 동결(freeze).(이 코드의 목적ㅇ느 Policy 모델 학습으로 Quality 모델은 이미 학습된 상태로 가져온 것이고, Policy 모델 학습 중에 Quality 모델까지 같이 바뀌면 안되니 동결)
- QUALITY_NODEL_PATH: "best_model_blaine.h5"
- quality_model.trainable = False: 이 모델의 가중치를 절대 바꾸지 않도록 고정


# Cell 8
"Xq_raw = np.concatenate([u_cur_raw, mon_raw, qprev_raw], axis=1)"
품질 모델의 입력값 38개를 하나의 배열로 합침
- np.concatenate: 여러 배열을 이어붙이는 함수
- axis=1: 옆으로(열 방향) 이어붙여라


# Cell 9
"train_ds = tf.data.Dataset.from_tensor_slices((Xp_train, yT_train)) \
    .shuffle(20000, seed=RANDOM_STATE)\
    .batch(BATCH_SIZE)\
    .prefetch(tf.data.AUTOTUNE)
numpy 배열을 TensorFlow가 학습에 사용할 수 있는 Dataset 형태로 변환
- tf.data.Dataset.from_tensor_slices((Xp_train, yT_train)): 입력-정답 쌍으로 묶어서 Dataset 생성
- .shuffle(20000, seed=RANDOM_STATE): 20000개씩 섞어서 학습 순설르 랜덤화
- .batch(BATCH_SIZE): 1024개씩 묶어서 한 번에 학습
- .prefetch(tf.data.AUTOTUNE): 학습하는 동안 다음 배치를 미리 준비하여 속도 향상
- val, test는 평가용이라서 순서가 바뀌면 안됨. 


# Cell 10
"policy_net = build_policy_model((POLICY_INPUT_DIM,))"
위에서 정의한 build_policy_model 함수를 실제로 호출해서 모델을 만듦
- (POLICY_INPUT_DIM,): 입력 shape을 전달. (39,) - 원소가 하나인 튜플(shape 표현)