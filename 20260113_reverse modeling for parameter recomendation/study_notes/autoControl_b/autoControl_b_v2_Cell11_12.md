# Cell 11
"DELTA_MAX_TF = tf.constant(DELTA_MAX.reshape(1,-1), dtype=tf.float32)"
numpy 배열인 DELTA_MAX를 TensorFlow에서 쓸 수 있는 상수로 변환
- DELTA_MAX.reshape(1, -1): shape(9,)를 (1, 9)로 변환
- tf.constant: TensorFlow 상수로 변환. numpy 배열과 달리 학습 중에 절대 바뀌지 않는 값으로 고정(TensorFlow는 TensorFlow 형식의 데이터끼리만 연산함. numpy 배열을 그대로 쓰면 내부저긍로 자동 변환이 일어나지만, tf.constant로 미리 변환해두면 더 안정적)


# Cell 12
"class ControlRecommender(tf.keras.Model):
    def __init__(self, policy_net: tf.keras.Model,  quality_model: tf.keras.Model, lam_l2: float = 1.0, lam_l1: float = 0.1, huber_delta: float = 1.0):
        super().__init__()
        self.policy_net = policy_net
        self.quality_model = quality_model

        self.lam_l2 = lam_l2
        self.lam_l1 = lam_l1
        self.huber_delta = huber_delta

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pred_loss_tracker = tf.keras.metrics.Mean(name="pred_loss")
        ....
- class ControlRecommender(tf.keras.Model): ControlRecommender라는 새로운 클래스 생성. tf.keras.Model 상속(tf.keras.Model이 가진 기능을 그대로 물려받으면서 우리만의 기능 추가)
- __init__: 객체가 만들어진 직후 자동으로 실행되는 특별한 메서드. 객체를 처음 만들 때 필요한 정보를 넣고 준비하는 곳. 클래스를 만들 때 한 번만 실행
- super().__init__(): 부모인 tf.keras.Model이 원래 가지고 있던 기본 준비 작업 먼저 실행(keras 모델로서 필요한 기본 설정을 먼저 해주는 것, 부모 클래스의 __init__ 매서드 실행)
- self: 현재 만들어지는 객체 자신(여기서는 ControlRecommender). (객체 정보와 상관없는 기능이라면 @staticmethod를 쓰고 self 없이 만들 수 있음)
- self.loss_tracker = tf.keras.metrics.Mean(name="loss"): 최종 손실값
- self.pred_loss_tracker = tf.keras.metrics.Mean(name="pred_loss"): 목표 품질과 예측 품질 사이의 오차
- self.delta_l2_tracker = tf.keras.metrics.Mean(name="delta_l2"): 조절값 변경량의 제곱 기준 크기 (큰 제어 변화량 정도)
- delta_l1_tracker = tf.keras.metrics.Mean(name="delta_l1"): 조절값 변경량의 절댓값 기준 크기 (전체 제어 변화량 정도)

"@property
def metrics(self):
    return [self.loss_tracker, self.pred_loss_tracker, self.delta_l2_tracker, self.delta_l1_tracker]"
학습할 때 이 네 가지 값을 기록하고 보여줘. TensorFlow에게 이 클래스가 추적하는 지표 4개야라고 알려주는 함수
- @property: 이 함수를 변수처럼 접근할 수 있게 해주는 데코레이터. 어떤 동작을 시키는 함수라기보다, 객체가 가진 정보처럼 다뤄야 하는 값이다라고 표현하는 방법 (이 모델이 현재 가지고 있는 정보처럼 보이게 하기 위해 사용. model.metrics() 대신 model.metrics로 접근 가능)
- def metrics: Tensorflow가 학습/평가 시 자동으로 호출하는 특수 함수(Keras가 정해둔 Model.metrics라는 속성을 재정의. 즉 함수가 아닌 속성을 접근)
- return [...]: 4개의 추적 도구를 리스트로 반환. Tensorflow가 이걸 보고 자동으로 초기화하고 출력

"def call(self, Xp_batch, training=False):
....
"
데이터가 들어왔을 때 Policy 모델로 제어값 변화량을 계산(추천)하고, Quality 모델로 새 Blaine을 예측하는 과정(계산 값으로 바꿨을 때 품질이 목표에 가까운지 확인) (추천과 품질 예측)
얼마나 바꾸라고 추천했는가?: delta
바꾼 뒤 조절값은 무엇인가?: u_new
그렇게 바꾸면 품질이 얼마로 예상되는가?: y_pred
- call: 클래스를 실제로 사용할 때마다 실행. 모델에 데이터를 넣었을 때 실제로 실행되는 함수. 입력 데이터를 넣었을 대 어떤 계산을 하지 정의하는 부분
- Xp_no_prev = Xp_batch[:, 0:39]: Policy 모델은 39개만 입력받으니까 마지막 qprev 제외
-  delta_raw = self.policy_net(Xp_no_prev, training=training): Policy 모델에 넣어서 delta_raw (-1~1) 출력(현재 조절값에서 더하거나 뺄 변화량). training=training은 지금 학습 중인지 평가 중인지 정책 모델에 전달하는 옵션(True: 학습, False: 평가)
- delta = delta_raw * DELTA_MAX_TF: Policy 모델의 출력값 -1,1을 실제 변화량 -0.2, 0.2로 스케일링
- u_new = tf.clip_by_value(u_cur + delta, 0.0, 1.0): 현재 조절값 u_cur에 delta를 더해서 만든 값. tf.clip_by_value: 지정한 범위를 벗어나면 잘라냄
- Xq_new = tf.concat([u_new, mon, qprev], axis=1): 새 제어값으로 Quality 모델 입력 38개 조합
- y_pred = self.quality_model(Xq_new, training=False): Quality 모델로 새 Blaine 예측. training=Flase는 quality 모델을 학습 모드가 아닌 예측 모드로 사용한다는 전달값
- return u_new, y_pred, delta: 새 제어값, 예측 Blaine, 변화량 반환

"def train_step(self, data):
        ...
"
한 배치를 실제로 학습시키는 전체 절차 (call()의 결과를 보고 policy_net을 조금 수정하는 부분)
- def train_step(self, data): 앞서 만든 tf.data.Dataset.from_tensor_slices((Xp_train, yT_train))를 data로 가지고 옴
- with tf.GradientTape() as tape: 아래 계산 과정 기록(최종 오차를 줄이려면 policy_net 안의 가중치를 어느 방향으로 얼마나 바꿔야 하는지 알아야 하고 그걸 계산하려면, loss가 만들어지기까지 어떤 계산을 거쳤는지 알아야 함)

"_, y_pred, delta = self(Xp_batch, training=True)"
 앞에서 만든 call()을 실행해서 결과 3개를 받아오는 코드 (_는 앞에서 u_new였으나, 이 부분에서는 사용하지 않으므로 저렇게 받음)

"pred_loss = tf.reduce_mean(
    tf.keras.losses.huber(y_target_batch, y_pred, delta=self.huber_delta))"
목표 품질 y_target_batch와 예상 품질 y_pred가 얼마나 다른지 huber loss로 계산해서, 그 평균을 pred_loss에 저장 (목표 품질과 예상 품질 사이의 평균 오차)

"delta_l2 = tf.reduce_mean(tf.reduce_sum(tf.square(delta), axis=1))"
추천한 조절값 변화량 delta가 너무 큰지 계산해서, 그 정도를 delta_l2에 저장
- delta: 조절값 9개를 각각 얼마나 바꾸라고 추천한 값
- tf.square(delta): 이 코드에서는 증가/감소는 중요하지 않고, 얼마나 크게 바꾸려고 했는가를 보기 위해 제곱(큰 변화량을 더 강하게 보기 위해. 조절값 하나를 아주 크게 바꾸는 추천은 특히 싫어하겠다는 의미)
- tf.reduce_sum(tf.square(delta), axis=1): 제곱값을 한 데이터 안에서 전부 더함. delta의 모양은 (배치 안 데이터 개수, 조절값 9개)니까, axis=1은 각 데이터마다 조절값 9개를 더하라는 뜻
- tf.reduce_mean(...): batch 안의 여러 데이터에서 나온 값을 평균

"delta_l1..."
l2처럼 조절값을 너무 많이 바꾸지 않게 하기 위한 값으로 절댓값 사용

"loss = pred_loss + self.lam_l2 * delta_l2 + self.lam_l1 * delta_l1"
앞서 계산한 세 가지 값을 합쳐서 모델이 줄여야 할 최종 점수 loss를 만드는 부분
- pred_loss: 목표 품질과 예상 품질의 차이
- delta_l2: 특정 제어값을 너무 크게 바꾼 정도
- delta_l1: 제어값들을 전체적으로 많이 바꾼 정도

"grads = tape.gradient(loss, self.policy_net.trainable_variables)"
지금 계산한 loss를 줄이려면, policy_net 안의 가중치들을 각각 어떻게 바꿔야 하는지 계산해서 grads에 저장 (실제로 가중치를 줄이는 것은 아니고 어떻게 바꿔야 하는지 계산만 해서 저장)
- self.policy_net.trainable_variables: policy_net 안에서 학습하면서 바꿀 수 있는 값들 (예로 weight, bias 등)
- tape.gradient(loss, self.policy_net.trainable_variables): gradient는 변화 방향을 알려주는 값으로 양수면 가중치를 줄이는 방향, 음수면 가중치를 늘리는 방향

"self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))"
앞서 계산한 grads를 이용해 policy_net의 가중치를 실제로 바꾸는 부분
- zip(): 두 묶음의 값을 같은 순서끼리 한 쌍으로 묶는 것
- optimizer: 모델을 만들 때 설정한 방법 이용해 가중치 수정
- apply_gradients(): gradient 정보를 가중치에 적용

"self.loss_tracker.update_state(loss)"
이번 배치에서 계산한 최종 loss 값을 loss_tracker에 기록 (학습 결과를 로그에 보여주기 위해 loss를 기록하는 역할)
- tf.keras.metrics.Mean: 여러 번 들어오는 숫자의 평균을 기록하는 도구

"return {m.name: m.result() for m in self.metrics}"
지금까지 기록한 지표들의 이름과 평균값을 묶어서 keras에 돌려줌(딕셔너리)

