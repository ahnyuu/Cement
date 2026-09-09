# Cell 4 - 배열 만들기, 데이터 분배
 "X_raw = df[X_COLS].to_numpy(dtype = np.float32)"
 - df[X_COLS]: 표에서 38개 입력 열만 선택
 - .to_numpy(): 표 형식을 숫자 배열(행렬)로 변환
 - dtype=np.float32: 숫자 형식을 32비트 실수로 지정
 - 결과: 60,572행 * 38열 행렬

 "y_raw = df[[Y_COL]].to_numpy(dtype=np.float32)"
 - blaine 컬럼만 꺼내 숫자 배열로 변환
 - 결과: 60,572행 * 1열 배열

 "print("X_raw shape:", X_raw.shape)
 print(y_raw shape:", y_raw.shape)"
 배열 크기 확인. shape은 (행 수, 열 수) 보여줌

 "train_x_raw, temp_x_raw, rain_y_raw, temp_y_raw = train_test_split(
    X_raw, y_raw, test_size=0.30, random_state=42)"
60,572건을 무작위로 둘로 나눔
 - test_size=0.30: 30%를 임시(temp)로, 70%를 훈련으로 (test_size는 항상 두 번째 변수의 비율을 뜻함)
 - random_state=42: 무작위 순서를 고정. 랜덤으로 나누긴 하지만, 랜덤 결과가 매번 같음 (42는 관례적으로 많이 쓰는 숫자, 의미 없음)

 "val_x_raw, test_x_raw, val_y_raw, test_y_raw = train_test_split(
temp_x_raw, temp_y_raw, test_size=0.50, random_state=42)"
위와 같음

"print("Split shapes:",
"train", train_x_raw.shap, train_y_raw.shape,
"| val", val_x_raw.shape, val_y_raw.shape,
"|test", test_x_raw.shpae, test_y_raw.shape)"
Split shapes: train(행, 열) (행, 열) |val(행, 열) (행, 열) | test(행, 열) (행, 열)


# Cell 5 
"x_scaler = MinMaxScaler(feature_range=(0.0, 1.0))"
feature_range=(0.0, 1.0): 나중에 변환할 때 모든 값을 0.0~1.0 범위를 사용하겠다는 설정값 지정 (도구 + 기억장치)

"x_scaler.fit(X_raw)"
 - fit: 전체 데이터를 보고 각 변수의 최솟값과 최댓값을 계산해서 기억
 - 정보를 x_scaler에 저장

"blaine_prev_idx = X_COLS.index("blaine_prev")"
 - X_COLS.index("blaine_prev"): 리스트에서 "blaine_prev"가 몇 번째 위치에 있는지 찾는 명령
 - X_COLS[0] = "RP_roller_p1"...
   X_COLS[37]= "blaine_prev"
   결과적으로 blaine_prev_idx=37 저장
   -> 이 번호를 통해 x_scaler가 38개 변수 전부의 최소/최대값을 배열로 저장하는데, blaine_prev의 값만 따로 수정하려면 38개 중 몇 번째냐를 알아야 하기 때문 (fit이 자동으로 계산한 값에 오차가 있어, 강제로 수정하기 위함)

   "for idx in [blaine_prev_idx]:"
 - [blaine_prev_idx] = [37] = 1번 반복 (리스트 하나)
 - idx = 37 (입력)

 "x_scaler.data_min_[idx] = CUSTOM_MIN"
 x_scaler 안에 저장된 38개 변수의 최솟값 배열 중 37번째(blaine_prev) 값만 3030으로 강제 수정

 "x_scaler.data_range_[idx] = CUSTOM_MAX - CUSTOM_MIN
 x_scaler.scale_[idx] = 1.0 / (CUSTOM_MAX - CUSTOM_MIN)
 x_scaler.min_[idx] = - CUSTOM_MIN / (CUSTOM_MAX - CUSTOM_MIN)"
 - data_range: 최댓값 - 최솟값 = 2470
 - scale: 1을 범위로 나눈 값 (1/2470)
 - min_: 정규화 공식 절편 (-3030/2470)
 fit이 자동 계산한 관련 값들도 전부 3030~5500 기준으로 맞추는 것. 이 3가지가 일치해야 transform이 올바르게 작동

 "blaine_idx = Y_COL.index("blaine")"
 blaine_idx = 0

 "train_x = x_scaler.transform(train_x_raw).astype(np.float32)"
 - x_scaler.transform(train_x_raw): fit으로 기억해둔 기준으로 train_x_raw의 모든 값을 0~1로 변환
 - .astype(np.float32): 변환된 값들의 숫자 형식을 32비트 실수로 통일

 "joblib.dump(x_scaler, "quality_x_scaler_38.pkl")"
 - fit하고 커스텀 값까지 수정한 x_scaler, y_scaler 파일로 저장
 - joblib.dump: 파이썬 객체를 파일로 저장하는 함수. joblib.dump(저장할 객체, 저장할 파일명)