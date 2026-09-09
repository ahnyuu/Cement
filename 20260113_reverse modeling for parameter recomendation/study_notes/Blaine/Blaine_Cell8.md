 # Cell 8
 "history = model.fit(
    train_x, train_y,
    epochs=1000,
    batch_size=1024,
    validation_data=(val_x, val_y),
    callbacks=[early_stopping, 촏차ㅔㅐㅑㅜㅅ (이거 한번 확인하고 PPT 넣기)]
 )"
 실제 학습을 실행하는 부분 (model.fit - 실제 학습)
 - train_x, train_y: train_x를 입력으로 넣어 모델을 돌리고 나온 결과인 예측값과 train_y를 비교
 - epochs=1000: 최대 1000번 반복 학습
 - batch_size=1024: 한 번에 1024개씩 묶어서 학습
 - validation_data=(val_x, val_y): 매 에포크마다 검증 데이터로 val_loss 계산 (검증 데이터는 학습에 사용되는 게 아닌, 모델의 정확도 확인용)
 - callbacks=[early_stopping, checkpoint]: 앞에서 만든 두 설정을 학습 중에 사용
 - history: 학습 과정(loss, val_loss 변화) 을 저장

 "best_model = tf.keras.models.load_model('best_model_blaine.h5')
 test_loss = best_model.evaluate(test_x, test_y)
 print(f'Test Loss (Best Model): {test_loss}')"
 - load_model(): checkpoint가 저장한 가장 좋은 모델 호출
 - best_model.evaluate(test_x, test_y): 한 번도 본 적 없는 테스트 데이터로 최종 성능 측정

 "plt.plot(history.history['loss'], label='Training Loss')"
 - history.history['loss']: 매 에포크마다 기록된 train_loss 목록
 - plt.plot(): 꺾은선 그래프로 그림
 - plt.legend(): 어떤 선이 뭔지 범례 표시
 - plt.show(): 화면 출력


 # Cell 9
 "best_model = tf.keras.models.load_model('best_model_blaine.h5')"
 .h5: 모델 구조 + 가중치 + 옵티마이저 상태를 모두 담은 파일 형식

 "idx = min(1, test_x.shapee[0]-1)"
 샘플로 확인할 행 번호 지정. 둘 중 더 작은 값(작은 숫자) 선택 
 - test_x.shape[0]-1: [0]은 9086. 여기에 -1을 하여 9085. 즉 테스트 데이터의 마지막 인덱스. 

"sample_x = test_x[idx].reshape(1, -1)"
 - test_x[1]은 shape이 (38,)인 1차원 배열
 -best_model.predict()는 (샘플 수, 변수 수) 형태의 2차원 배열을 요구
 - .reshape(1, -1)로 (1, 38) 형태로 바꿔줌. -1은 나머지 차원을 자동 계산
 - 예를 들어, 원소 12개가 있을 때 arr.reshape(3, 4)를 하면 3행 4열이 됨. arr.shape(4, -1)을 하면 열은 몇 개인지 몰라도 자동으로 12가 되도록 계산해서 4행 3열이 됨

 "pred_y = best_model.predict(sample_x, verbose=0)"
 - best_model.predict(sample_x): 모델에 sample_x를 넣어서 예측값을 출력
 - verbose=0: 예측 중 진행상황을 터미널에 출력하지 않음
 - pred_y: 정규화된 예측값 (0~1)

 "actual_blaine = y_scaler.inverse_transform(sample_y)[0, 0]"
 정규화된 값을 원래 blaine 단위로 되돌림. 결과는 [[3530.0]] 같은 2차원 배열로 나옴

 "print(f"Abs error : {abs(actual_blaine - pred_blaine): .2f}")"
 - abs(): 절대값 함수


 # Cell 14 (랍니다...번호 무시해라)
 "test_x_arr = np.asarray(test_x)"
 test_x를 NumPy ndarray로 강제 변환. 간혹 np.matrix 형태로 되어 있으면 이후 연산에서 오류가 날 수 있어 안전하게 변환 (이미 ndahrray면 그대로 둠)

 "y_pred_norm = np.asarray(best_model.pedict(test_x_arr))
 y_true_norm = test_y_arr"
 - best_model.predict(test_x_arr): 9086개 샘플 전체를 모델에 넣고 예측값을 한꺼번에 구함
 - 출력 shape: (9086, 1)이고 값은 MinMaxScaler로 정규화된 0~1 사이 실수

 
 "y_pred_real = y_scaler.inverse_transform(y_pred_norm)"
 0~1 사이 정규화 값을 원래 blaine 단위로되 되돌림. 커스텀 범위를 3030~5500으로 설정했기 때문에 역변환도 그 범위 기준으로 계산. 결과 shpae은 동일하고 값은 실제 blaine 값

 "abs_err = np.abs(y_ture_real - y_pred_real)"
 - y_ture_real - y_pred_real은 9086개 샘플 각각의 오차를 계산. 예측이 실제보다 크면 음수, 작으면 양수
 - np.abs()로 모두 양수(절댓값)으로 만듦

 "mae = mean_absolute_error(y_ture_real, y_pred_real)
 rmse = np.sqrt(mean_squared_error(y_ture_real, y_pred_real))
 r2 = r2_score(y_ture_real, y_pred_real)"
 - mean_absolute_error: 9086개 절대오차의 평균. 결과 35.63은 평균적으로 실제 blaine과 예측 blaine이 35.63 차이난다는 의미
 - mean_squared_error: 오차를 제곱한 뒤 평균. 여기에 np.sqrt()로 루트를 씌워 RMSE 계산. 결과 54.57은 MAE보다 큰데, 큰 오차에 제곱을 하기 때문에 그 영향이 더 크게 반영
 - r2_score: 모델이 blaine 값의 변동을 얼마나 설명하는지 나타냄. 0.7883은 약 78.83%를 설명한다는 의미이고 1.0이 완벽한 예측, 0.0이 평균값만 예측하는 것과 동일한 수준

 "min_err = float(np.min(abs_err))
 max_err = float(np.max(abs_err))"
 9086개의 절대오차 중 가장 작은 값과 가장 큰 값을 구함. float()으로 감싼 이유는 np.min()의 반환값이 NumPy 고유 타입(np.float32)인데, 이를 Python 기본 실수 타입으로 변환하기 위해


 # Cell 16 (마지막)
 "actual_blaine = y_scaler.inverse_transform(y_ture_norm)[;, 0]
 predicted_blaine = y_scaler.inverse_transform(y_pred_norm)[;,0]"
 정규화된 값을 실제 blaine 단위로 역변환. inverse_transform() 결과가 (9086, 1) 2차원 배열이므로 [;, 0]으로 (9086,) 1차원 배열로 변환. 
 - [;, 0]: 모든 행의 0번 열만 가져와라
 - categorize_blaine(): 값을 하나씩 꺼내야 하기 때문에 1차원으로 만드는 것

 "def categorize_blaine(blaine_values):
      return np.array([
         "OK" if 3700 < = value <= 3900 else "NG"
         for value in blaine_values
      ])"
 blaine 값이 3700 이상 3900 이하면 OK, 그 외에는 NG로 분류하는 함수. for value in blaine_values로 9086개 값을 하나씩 꺼내서 판별하고, np.array()로 결과를 배열로 만듦. 결과는 ["NG", "OK", "NG",....] 같은 shape (9086,) 배열

 "conf_matrix = confusion_matrix(
   actual_labels,
   predicted_labels,
   labels=["OK", "NG"]
 )"
 sklearn의 confusion_matrix()로 혼동 행렬을 계산. labels=["OK", "NG"]로 행/열 순서를 OK가 먼저 오도록 지정

 "df_conf_matrix = pd.DataFrame(
   conf_matrix,
   index = ["Actual Spec_In", "Actual Spec_Out"],
   columns = ["Predicted Spec_In", "Predicted Spec_Out"]
 )"
 2x2 배열인 conf_matrix를 Pandas DataFrame으로 변환. index는 행 이름, columns는 열 이름.

 "plt.figure(figsize=(8, 6))"
 가로 8인치, 세로 6인치 크기의 그래프 틀 생성

"ax = sns.heatmap(
   df_conf_matrix,
   annot=true,
   fmt="d",
   cmap="Blues",
   linewidths=1,
   linecolor="white"
)"
- sns.heatmap(): 혼동 행렬을 색상 히트맵으로 그림
- annot=True: 각 칸 안에 숫자를 표시
- fmt="d": 숫자를 정수로 표시
- cmap="Blues": 파란색 계열 색상 사용
- linewidths=1, linecolor="white": 각 칸 사이에 흰색 구분선을 1픽셀로 그려라
