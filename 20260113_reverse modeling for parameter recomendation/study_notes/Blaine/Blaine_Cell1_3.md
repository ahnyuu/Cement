# Cell 1 - 라이브러리 불러오기

"import numpy as np"
숫자 배열을 다루는 도구 numpy를 불러오고, 앞으로는 np라고 명명

"import matplotlib.pyplot as plt"
그래프를 그리는 도구 pyplot을 불러오고, 앞으로는 plt로 명명. 이후 학습 손실 그래프를 그릴 때 활용

"%matplotlib inline"
%로 시작하는 건 Jupyter Notebook 전용 명령어. 그래프를 노트북 셀 안에 보여주라는 뜻

"#tf.compat.v1.disable_eager_execution()"
과거 버전 Tensorflow용 코드. 현재는 Tensorflow v2를 사용하고 있지만, 이 코드는 TF v1으로 동작하게 하는 코드. TF v1 동작 방식이 필요한 경우 사용

"from sklearn.model_selection import train_test_split"
사이킷런에서 train_test_split이라는 함수만 호출. 데이터를 나눌 때 사용

"from sklearn.preprocessing import MinMaxScaler"
사이킷런에서 MinMaxScaler라는 정규화 도구 호출

"import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2""
운영체제 관리 기능. environ은 환경변수 설정으로, 2로 설정하면 TF가 출력하는 불필요한 경고 메시지를 줄임

"import pandas as pd"
CSV 파일을 다루는 도구로, 데이터를 엑셀처럼 읽고 다룰 때 사용. 이후 pd로 명명

"from tensorflow import keras"
딥러닝 프레임워크인 TF에서 keras를 호출. 이는 신경망을 쉽게 만들고 학습시키는 핵심 도구. 아래와 같은 신경망의 구성요소들을 가져옴
 - Dense: 뉴런들이 촘촘히 연결된 층
 - BatchNormalization: 학습을 안정시키는 층
 - Dropout: 과적합 방지를 위해 뉴런 일부를 무작위로 끄는 층

"import seaborn as sns"
고급 그래프 도구. 이후 cell 16에서 Confusion Matrix를 시각화 할 때 이용

"from sklearn.metrics import confusion_matrix, accuracy_score"
모델 성능을 평가하는 함수들. cell 16에서 OK / NG 예측이 얼마나 정확한지 계산할 때 사용

"from tensorflow.keras.layers import LeakReLU"
신경망 층의 활성화 함수 중 하나인 LeakyReLU를 호출. 그러나 이 코드에서는 실제로 사용하지 않음

"from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Add, Dropout"
신경망을 구성하는 층 종류를 가져옴
 - Input: 입력층 (38개 변수가 들어오는 문)
 - Dense: 완전연결층 (모든 뉴런이 다음 층과 연결)
 - BatchNormalization: 각 층의 출력값을 안정화시키는 층
 - ReLU, Add: 실제로 사용하지 않음
 - Dropout: 학습 중 뉴런 일부를 무작위로 꺼서 과적합 방지


"from tensorflow.keras.models import Model"
복잡한 구조의 신경망을 만들 때 사용하는 방식. 그러나 이 코드에서는 사용하지 않음

"from tensorflow.keras.models import Sequential"
층을 순서대로 쌓는 방식으로 신경망을 만드는 도구
 - Model: 더 자유롭게 모델을 만들 때 사용. 보통 function API 방식이라고 함.
 - Sequential: 층을 순서대로 일렬로 쌓는 모델을 만들 때 사용. 앞 층의 출력이 바로 다음 층의 입력으로 들어가는 단순한 구조

 "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint"
 학습 중 자동으로 동작하는 기능
  - EarlyStopping: 검증 손실이 300번 연속 개선되지 않으면 학습 자동 중단
  - ModelCheckpoint: 검증 손실이 가장 낮은 시점의 모델을 파일로 자동 저장

  "import joblib"
  파이썬 객체를 파일로 저장 / 불러오는 도구. cell 5에서 정규화 정보(scaler)를 .pkl 파일로 저장할 때 사용

  "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'"
  앞에서 2로 설정한 것을 3으로 덮어씀. (숫자가 클수록 TF의 경고 / 정보 메시지를 더 많이 숨김)

 
  # Cell 2 - 데이터 불러오기 (정규화 전, CSV 파일을 읽어 df에 저장)
"directory = "database/""
데이터 파일이 있는 폴더 경로를 directory라는 변수에 저장. 실제 위치는 database/ 폴더

"raw_csv = "cmall_remove_duplicate_m_all.csv"
불러올 파일 이름을 raw_csv라는 변수에 저장.
 - cmall: cement all (시멘트 전체 데이터)
 - remove_duplicate: 중복 행 제거
 - m_all: 분(minute) 단이 전체 데이터

"df = pd.read_csv(os.path.join(directory, raw_csv))"
 - os.path.join(directory, raw_csv): database / cmall_remove_duplicate_m_all.csv 경로 생성
 - pd.read_csv(): 그 경로의 CSV 파일을 읽어 DataFrame 형태로 변환


 # Cell 3 - 입력 / 출력 변수 정의 (X, Y 정의 및 결측치 행 제거)
 "X_COLS = [....]"
 입력 변수 38개의 이름을 리스트로 정의 (크게 제어변수(조작 가능), 모니터링(측정만 가능), 이전 Blaine으로 구성)

 "Y_COL = "blaine""
 출력 변수는 blaine 한 개

 "assert len(X_COL) == 38, f"X_COLS length must be 38, got {len(X_COLS)}""
 assert는 이 조건이 틀리면 즉시 오류를 내라는 검증 명령. len(X_COLS)는 리스트 길이 측정. 실수로 변수를 하나 빠뜨리거나 추가했을 때 바로 잡아내기 위한 안전장치

 "df = df.dropna(subset = X_COLS + [Y_COL]).reset_index(drop = True)"
 - dropna(subset = X_COLS + [Y_COL]): 38개 입력 변수나 blaine 값 중 하나라도 비어있는(결측) 행 삭제
 - .reset_index(drop = True): 인덱스를 0부터 다시 매김. 삭제 후 행 번호에 구멍이 생기는 것을 방지