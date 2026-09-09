# Cell 3
"assert len(CONTROL_COLS_9)==9"
- assert: 이게 맞는지 확인하고 틀리면 에러를 내라

"QUALITY_INPUT_COLS = CONTROL_COL_9 + MONITOR_COLS_28 + [QUALITY_PREV_COL]


# Cell 4
"if TARGET_QUALITY_COL not in df.columns:
    df[TARGET_QUALITY_COL] = TARGET_BLAINE_DEFAULT"
blaine_target 열이 df 열에 없으면, 60572개 행 전부 3800.0을 채워 새로운 열을 만들어라

"required_cols = list(set(
    QUALITY_INPUT_COLS +
    [POLICY_CUR_COL] +
    [TARGET_QUALITY_COL]
))"
셋을 합치면 중복이 생기니 set()으로 중복을 제거하고, 다시 list()로 리스트로 변환

"missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")"
필수 열 중에서 csv에 없는 열이 있는지 찾고 대응하는 코드. 
- required_cols 목록에 있는 열(c) 중에서, df에 없는 것만 골라서 missing_cols에 저장해라
- for c in required_cols: required_cols에서 항목을 하나씩 꺼내서 c라고 부름
- if c not in df.columns: df에 없는 c만 골라라
- c:  c를 결과 리스트에 추가
- missing_cols에 뭔가 있으면 에러를 내고 프로그램을 멈춤. 어떤 열이 없는지 이름도 알려줌

"df = df.dropna(subset=required_cols).reset_index(drop=True)"
필수 열(required_cols) 중 빈 값이 있는 행을 통째로 삭제
- .dropna(): 빈 값이 있는 행을 떨어뜨리는(drop) 함수
- subset=required_cols: 모든 열이 아니라 required_cols에 있는 열만 체크해서 떨어뜨려라(이걸 안 하면 모든 열에 빈 값이 있는 행 모두 삭제)
- .reset_index(): 행 번호를 0부터 다시 매김
- drop=True: 기존 행 번호는 버림 (drop=False는 기존 번호가 새 열로 저장)


# Cell 5 - 60572개 데이터를 훈련/검증/테스트 세트로 쪼개는 셀
"idx_all = np.arange(len(df))"
행 번호 배열을 만듦
- np.arange: 0부터 ()안 값-1까지 정수 배열 생성

"idx_train, idx_temp = train_test_split(idx_all, test_size=0.30, random_state=RAMDOM_STATE)"
행 번호를 70%, 30%로 랜덤하게 쪼갬. train은 70%, temp는 30%

"idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=RANDOM_STATE)"
전체 30%의 temp를 다시 50%로 나눠 각각 전체의 15%의 val, test 세트 생성