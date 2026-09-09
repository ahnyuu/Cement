# Cell 15
"print("Policy output shape:", policy_net_loaded.output_shape)"
policy_net_loaded 모델의 출력 형태 확인. (None, 9)
- None: 한 번에 몇 개의 데이터를 넣을지는 정해져 있지 않음
- 9: 데이터 한 개마다 결과를 9개 출력


# Cell 16
"err_before_norm = []"
오차를 저장할 빈 리스트 생성. 비교해야 할 결과가 네 종류라서 리스트도 네 개
- before_norm: 정책 적용 전, 정규화된 범위에서의 오차
- after_real: 정책 적용 후, 실제 Blaine 단위 오차

"for Xp_batch, y_target_norm in dataset.skip(n_skip_batches).take(n_batches):"
데이터셋에서 정한 만큼의 배치를 하나씩 꺼내는 반복문

"delta = delta_raw * DELTA_MAX_TF"
policy net이 낸 delta_raw를 실제로 허용된 변화 범위에 맞게 조정하는 코드
- 앞서 나온 delta_raw는 제어변수 9개에 대한 변화 방향과 비율이 들어있음. 또한 tanh 때문에 -1~1 범위로 "얼마나 세게 바꿀지의 비율"로 표현
- DELTA_MAX_TF: 각 제어변수를 실제로 바꿀 수 있는 최대량. 같은 위치끼리 곱함(delta_raw의 1번 값 * DELTA_MAX_TF의 1번째 값)

"u_new = tf.clip_by_value(u_cur + delta, 0.0, 1.0)"
현재 제어값에 추천 변화량을 더한 뒤, 결과가 0~1 범위를 벗어나지 않게 제한하는 코드
- u_cur + delta: 현재 제어값에 policy net이 추천한 변화량 더함
