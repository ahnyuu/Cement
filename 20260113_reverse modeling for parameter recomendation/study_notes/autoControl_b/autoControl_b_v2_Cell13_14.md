# Cell 13
"quality_model.trainable = False"
이미 만들어 둔 품질 예측 모델(quality_model)은 고정하고, 새로 만드는 정책 모델(policy)만 학습

"recommender.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR))"
recommender를 학습할 준비 상태로 설정하는 코드. 학습 중 가중치를 어떻게 수정할지를 정하는 optimizer로 Adam을 사용하겠다고 지정
- learning_rate=LR: 가중치를 한 번 수정할 때 얼마나 움직일지 정하는 학습률을 앞에서 만든 LR값으로 설정

"early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True
)"
- monitor="val_loss": 학습이 잘되고 있는지 판단할 기주능로 검증 데이터의 loss를 보겠다는 뜻
- patience=PATIENCE: val_loss가 좋아지지 않아도 바로 멈추지는 않고, PATIENCE에서 정한 횟수만큼 더 기다리겠다는 뜻
- restore_best_weights=True: 학습을 멈출 때, 마지막 가중치를 쓰는 게 아니라 검증 loss가 가장 좋았던 시점의 가중치로 되돌리겠다는 뜻

"history = recommender.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callback=[early_stop],
    verbose=1
)"
recommender를 실제로 학습시키고, 학습 결과를 history에 저장 (매 epoch의 기록을 저장)

"recommender.evaluate(test_ds, verbose=1)"
학습이 끝난 recommender를 가지고 한 번도 학습에 쓰지 않은 테스트 데이터 test_ds로 최종 점검하는 단계 (가중치를 수정하지 않음).
TensorFlow의 evaluate()가 test_ds를 배치 단위로 꺼내서, ControlRecommender에 직접 만든 test_step() 실행
- evaluate()가 test_step()을 호출한다는 직접적인 코드는 없지만, TensorFlow 내부의 tf.keras.Model 코드에 둘을 연결하는 코드가 들어있음

"policy_net.save(POLICY_SAVE_PATH)"
학습이 끝난 policy_net만 파일로 저장(recommender 전체가 아닌 policy net만 저장) (POLICY_SAVE_PATH = "best_policy_net.keras라는 이름의 파일로 저장")


# Cell 14
"hist = history.history"
history는 학습 기록을 담은 객체로 그 안에는 history.history(loss, val_loss 등 실제 기록값), history.epoch(실제 실행된 epoch 번호), history.params(batch 수, epoch 설정 값은 정보)가 전부 들어있음. 그래서 학습 결과 숫자를 그래프로 보기 위해 기록값만 꺼내 hist에 저장

"plt.figure(figsize=(10,5))"
새 그래프 만들기
- plt.figure(): 그래프를 그릴 빈 공간 생성
- figsize=(10,5): 그래프의 크기를 가로 10, 세로 5 정도로 정함

"plt.plot(hist["loss"], label="train total loss")
hist["loss"]에 저장된 값을 선으로 그림 (hist["loss"]에는 epoch마다 계산된 학습 데이터의 전체 loss가 들어있음). loss가 없으면 오류
- label은 이 선의 이름표
- 아래 나오는 get과 비교하면 loss는 학습에 반드시 있어야 하는 기록이라 없으면 오류가 나도록 hist[]를 사용했고, pred_loss는 추가로 기록한 세부 항목이라 없어도 동작이 되도록 hist.get() 사용

"plt.plot(hist["val_loss"], label="val total loss")"
hist["val_loss"] 값을 같은 그래프에 두 번째 선으로 그림

"plt.plot(hist.get("val_pred_loss", []), label="val pred_loss")"
- hist.get(): hist 안에 ...가 있으면 그 값을 꺼내고, 없으면 빈 리스트 출력. (없다고 오류가 나지는 않음)