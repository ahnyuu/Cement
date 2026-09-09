
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf

# -----------------------------
# 0) USER CONFIG (경로/스펙) - "고정" (요구사항)
# -----------------------------
DATA_DIR  = "database"
EVAL_FILE = "small_cases.csv"

QUALITY_RES_PATH  = "best_model_residue.h5"
POLICY_RES_PATH   = "policy_residue.keras"
SCALER_X_RES_PATH = "scaler_x_res_38.pkl"
SCALER_Y_RES_PATH = "scaler_y_res_1.pkl"

QUALITY_BLA_PATH  = "best_model_blaine.h5"
POLICY_BLA_PATH   = "policy_blaine.keras"
SCALER_X_BLA_PATH = "scaler_x_bla_38.pkl"
SCALER_Y_BLA_PATH = "scaler_y_bla_1.pkl"

# spec (표시/판정용)
RES_MIN, RES_MAX = 7.0, 9.0
BLA_MIN, BLA_MAX = 3700.0, 3900.0

# input columns (학습과 동일해야 함)
CONTROL_COLS_9 = [
    "RP_roller_p1","RP_roller_p2","RP_sep_rpm","RP_sep_fan_damper","RP_sep_BF_damper",
    "mill_BF_damper","mill_sep_rpm","mill_sep_fan_damper","grind_aid",
]
MONITOR_COLS_28 = [
    "RP_spac","RP_skew","RP_roller_energy1","RP_roller_energy2","RP_roller_vib",
    "RP_BE_energy1","RP_BE_energy2","RP_sep_BF_pressure","dosing_BE_energy","mill_feed_c",
    "mill_feed_cir","mill_energy","mill_in_temp","mill_out_temp","mill_BE_energy",
    "mill_out_gas_temp","mill_out_mater_temp","mill_BF_pressure","mill_sep_BF_pressure",
    "mill_sep_fan_damper","final_BE_1","final_BE_2","final_mater_temp",
    "feed_clinker","feed_gypsum","feed_slag","feed_FA","feed_total",
]
RES_COL   = "residue"
RES_T_COL = "residue_target"
BLA_COL   = "blaine"
BLA_T_COL = "blaine_target"

POLICY_INPUT_DIM = 28 + 1 + 9 + 1  # 39

# -----------------------------
# 1) CORE HELPERS
# -----------------------------
def split_norm_from_scaler_x(u_raw_1x9, mon_raw_1x28, qprev_raw_1x1, scaler_x):
    Xq_raw  = np.concatenate([u_raw_1x9, mon_raw_1x28, qprev_raw_1x1], axis=1).astype(np.float32)  # (1,38)
    Xq_norm = scaler_x.transform(Xq_raw).astype(np.float32)
    u_norm     = Xq_norm[:, :9]
    mon_norm   = Xq_norm[:, 9:9+28]
    qprev_norm = Xq_norm[:, 9+28:9+28+1]
    return u_norm, mon_norm, qprev_norm, Xq_norm


def policy_delta_norm(policy_net, mon_norm, qcur_norm, u_cur_norm, yT_norm, delta_max_default):
    Xp_39 = np.concatenate([mon_norm, qcur_norm, u_cur_norm, yT_norm], axis=1).astype(np.float32)  # (1,39)
    assert Xp_39.shape[1] == POLICY_INPUT_DIM
    delta_raw = policy_net(Xp_39, training=False).numpy().astype(np.float32)  # (1,9) in [-1,1]
    DELTA_MAX = np.array([delta_max_default]*9, dtype=np.float32).reshape(1, -1)
    delta_norm = delta_raw * DELTA_MAX
    return delta_norm, delta_raw


def quality_forward_norm(quality_model, u_new_norm, mon_norm, qprev_norm):
    Xq_new = np.concatenate([u_new_norm, mon_norm, qprev_norm], axis=1).astype(np.float32)  # (1,38)
    y_norm = quality_model(Xq_new, training=False).numpy().astype(np.float32)               # (1,1)
    return y_norm, Xq_new


def alpha_rule_res_first(res_before, bla_before,
                         res_min=RES_MIN, res_max=RES_MAX,
                         alpha_when_res_out=0.1,
                         alpha_when_res_in=0.7):
    res_in = (res_min <= res_before <= res_max)
    return float(alpha_when_res_in if res_in else alpha_when_res_out)


def extract_minmax_from_minmaxscaler(scaler_x, control_cols, monitor_cols):
    cols = control_cols + monitor_cols + ["qprev"]
    if not (hasattr(scaler_x, "data_min_") and hasattr(scaler_x, "data_max_")):
        raise ValueError("scaler_x must be a fitted MinMaxScaler with data_min_/data_max_")
    if len(cols) != len(scaler_x.data_min_):
        raise ValueError(f"Expected {len(cols)} features, but scaler has {len(scaler_x.data_min_)}")

    data_min = scaler_x.data_min_.astype(float)
    data_max = scaler_x.data_max_.astype(float)

    control_limits = {c: (data_min[i], data_max[i]) for i, c in enumerate(control_cols)}
    monitor_limits = {c: (data_min[9+i], data_max[9+i]) for i, c in enumerate(monitor_cols)}
    qprev_limits   = (data_min[-1], data_max[-1])
    return control_limits, monitor_limits, qprev_limits


def apply_range_margin(limits: dict, margin_ratio: float = 0.02) -> dict:
    out = {}
    for k, (lo, hi) in limits.items():
        lo = float(lo); hi = float(hi)
        span = hi - lo
        if span <= 0:
            out[k] = (lo, hi)
        else:
            out[k] = (lo - margin_ratio * span, hi + margin_ratio * span)
    return out


def clip_controls_raw(u_raw_1x9, control_cols, control_limits):
    u = u_raw_1x9.copy().astype(np.float32)
    for i, c in enumerate(control_cols):
        lo, hi = control_limits[c]
        u[0, i] = np.clip(u[0, i], lo, hi)
    return u


# ---- 실행 차단 검증: controls + monitors 모두 범위 체크(±2% margin 기준)
def validate_payload_controls_only(payload, control_limits):
    errors = []
    for k in ["controls", "monitors", "quality", "targets"]:
        if k not in payload:
            errors.append(f"Missing top-level key: '{k}'")
            return errors

    # controls only: strict + range
    for c, (lo, hi) in control_limits.items():
        if c not in payload["controls"]:
            errors.append(f"Missing control: {c}")
            continue
        try:
            v = float(payload["controls"][c])
        except Exception:
            errors.append(f"Control not numeric: {c}={payload['controls'][c]}")
            continue
        if not (lo <= v <= hi):
            errors.append(f"Control out of range: {c}={v} (allowed {lo}~{hi})")

    # quality & targets numeric check (그대로 유지)
    for qk in [RES_COL, BLA_COL]:
        if qk not in payload["quality"]:
            errors.append(f"Missing quality: {qk}")
        else:
            try:
                float(payload["quality"][qk])
            except Exception:
                errors.append(f"Quality not numeric: {qk}={payload['quality'][qk]}")
    for tk in [RES_T_COL, BLA_T_COL]:
        if tk not in payload["targets"]:
            errors.append(f"Missing target: {tk}")
        else:
            try:
                float(payload["targets"][tk])
            except Exception:
                errors.append(f"Target not numeric: {tk}={payload['targets'][tk]}")

    # delta_max
    try:
        dm = float(payload.get("delta_max", None))
        if dm <= 0:
            errors.append(f"delta_max must be > 0, got {dm}")
    except Exception:
        errors.append("delta_max missing or not numeric")

    return errors

def recommend_1step_from_payload(payload: dict, *, bundle: dict):
    u_raw   = np.array([[payload["controls"][c] for c in CONTROL_COLS_9]], dtype=np.float32)
    mon_raw = np.array([[payload["monitors"][c] for c in MONITOR_COLS_28]], dtype=np.float32)

    res_qcur = np.array([[float(payload["quality"][RES_COL])]], dtype=np.float32)
    bla_qcur = np.array([[float(payload["quality"][BLA_COL])]], dtype=np.float32)
    yT_res   = np.array([[float(payload["targets"][RES_T_COL])]], dtype=np.float32)
    yT_bla   = np.array([[float(payload["targets"][BLA_T_COL])]], dtype=np.float32)

    delta_max = float(payload["delta_max"])

    # 요구사항: qprev = qcur
    res_qprev = res_qcur.copy()
    bla_qprev = bla_qcur.copy()

    # residue 우선 alpha 규칙
    a = float(alpha_rule_res_first(float(res_qcur[0,0]), float(bla_qcur[0,0])))
    a = float(np.clip(a, 0.0, 1.0))

    # residue policy
    u_norm_res, mon_norm_res, qprev_norm_res, _ = split_norm_from_scaler_x(
        u_raw, mon_raw, res_qprev, bundle["scaler_x_res"]
    )
    qcur_norm_res = bundle["scaler_y_res"].transform(res_qcur).astype(np.float32)
    yT_norm_res   = bundle["scaler_y_res"].transform(yT_res).astype(np.float32)
    delta_res_norm, _ = policy_delta_norm(
        bundle["policy_res"], mon_norm_res, qcur_norm_res, u_norm_res, yT_norm_res, delta_max
    )

    # blaine policy
    u_norm_bla, mon_norm_bla, qprev_norm_bla, _ = split_norm_from_scaler_x(
        u_raw, mon_raw, bla_qprev, bundle["scaler_x_bla"]
    )
    qcur_norm_bla = bundle["scaler_y_bla"].transform(bla_qcur).astype(np.float32)
    yT_norm_bla   = bundle["scaler_y_bla"].transform(yT_bla).astype(np.float32)
    delta_bla_norm, _ = policy_delta_norm(
        bundle["policy_bla"], mon_norm_bla, qcur_norm_bla, u_norm_bla, yT_norm_bla, delta_max
    )

    # mix delta
    delta_mix_norm = (1.0 - a) * delta_res_norm + a * delta_bla_norm

    # apply in norm space + clip [0,1]
    u_new_norm_res = np.clip(u_norm_res + delta_mix_norm, 0.0, 1.0)

    # inverse to raw controls (res scaler 기준)
    Xq_new_res_norm = np.concatenate([u_new_norm_res, mon_norm_res, qprev_norm_res], axis=1).astype(np.float32)
    Xq_new_res_raw  = bundle["scaler_x_res"].inverse_transform(Xq_new_res_norm).astype(np.float32)
    u_new_raw = Xq_new_res_raw[:, :9]

    # 요구사항: 추천 출력은 원본 min/max로 clip
    u_new_raw = clip_controls_raw(u_new_raw, CONTROL_COLS_9, bundle["CONTROL_LIMITS_ORIG"])

    # predict residue next
    res_next_norm, _ = quality_forward_norm(bundle["quality_res"], u_new_norm_res, mon_norm_res, qprev_norm_res)
    res_next = bundle["scaler_y_res"].inverse_transform(res_next_norm).astype(np.float32)

    # predict blaine next: blaine scaler space로 재정규화
    u_new_norm_bla, mon_norm_bla2, qprev_norm_bla2, _ = split_norm_from_scaler_x(
        u_new_raw, mon_raw, bla_qprev, bundle["scaler_x_bla"]
    )
    bla_next_norm, _ = quality_forward_norm(bundle["quality_bla"], u_new_norm_bla, mon_norm_bla2, qprev_norm_bla2)
    bla_next = bundle["scaler_y_bla"].inverse_transform(bla_next_norm).astype(np.float32)

    # pack outputs
    u_new_dict = {c: float(u_new_raw[0,i]) for i,c in enumerate(CONTROL_COLS_9)}
    delta_dict = {c: float(u_new_raw[0,i] - u_raw[0,i]) for i,c in enumerate(CONTROL_COLS_9)}

    res_cur = float(res_qcur[0,0]); res_nxt = float(res_next[0,0])
    bla_cur = float(bla_qcur[0,0]); bla_nxt = float(bla_next[0,0])

    out = {
        "u_new": u_new_dict,
        "delta": delta_dict,
        "pred": {
            "residue_cur": res_cur,
            "residue_next": res_nxt,
            "blaine_cur": bla_cur,
            "blaine_next": bla_nxt,
            "alpha": a,
            "delta_max": float(delta_max),
        },
        "spec": {
            "residue_cur_ok": bool(RES_MIN <= res_cur <= RES_MAX),
            "residue_next_ok": bool(RES_MIN <= res_nxt <= RES_MAX),
            "blaine_cur_ok": bool(BLA_MIN <= bla_cur <= BLA_MAX),
            "blaine_next_ok": bool(BLA_MIN <= bla_nxt <= BLA_MAX),
            "both_next_ok": bool((RES_MIN <= res_nxt <= RES_MAX) and (BLA_MIN <= bla_nxt <= BLA_MAX)),
        },
    }
    return out


# -----------------------------
# 2) LOAD MODELS ONCE (cache)
# -----------------------------
@st.cache_resource
def load_bundle():
    scaler_x_res = joblib.load(SCALER_X_RES_PATH)
    scaler_y_res = joblib.load(SCALER_Y_RES_PATH)
    quality_res  = tf.keras.models.load_model(QUALITY_RES_PATH); quality_res.trainable = False
    policy_res   = tf.keras.models.load_model(POLICY_RES_PATH);  policy_res.trainable  = False

    scaler_x_bla = joblib.load(SCALER_X_BLA_PATH)
    scaler_y_bla = joblib.load(SCALER_Y_BLA_PATH)
    quality_bla  = tf.keras.models.load_model(QUALITY_BLA_PATH); quality_bla.trainable = False
    policy_bla   = tf.keras.models.load_model(POLICY_BLA_PATH);  policy_bla.trainable  = False

    # limits from MinMaxScaler
    CONTROL_LIMITS_ORIG, MONITOR_LIMITS_ORIG, _ = extract_minmax_from_minmaxscaler(
        scaler_x_res, CONTROL_COLS_9, MONITOR_COLS_28
    )

    # 실행 차단 검증은 ±2% margin 적용 범위로
    CONTROL_LIMITS_M = apply_range_margin(CONTROL_LIMITS_ORIG, margin_ratio=0.02)
    MONITOR_LIMITS_M = apply_range_margin(MONITOR_LIMITS_ORIG, margin_ratio=0.02)

    return {
        "scaler_x_res": scaler_x_res, "scaler_y_res": scaler_y_res,
        "quality_res": quality_res, "policy_res": policy_res,
        "scaler_x_bla": scaler_x_bla, "scaler_y_bla": scaler_y_bla,
        "quality_bla": quality_bla, "policy_bla": policy_bla,
        "CONTROL_LIMITS_ORIG": CONTROL_LIMITS_ORIG,
        "CONTROL_LIMITS_M": CONTROL_LIMITS_M,
        "MONITOR_LIMITS_M": MONITOR_LIMITS_M,
    }


# -----------------------------
# 3) STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Cement 1-step Control Recommender", layout="wide")
st.title("시멘트 공정 1-step 제어변수 추천 UI (내부 데모)")

bundle = load_bundle()

st.sidebar.header("옵션")
delta_max = st.sidebar.slider("DELTA_MAX (공격성)", min_value=0.001, max_value=0.20, value=0.02, step=0.001)

st.subheader("1) CSV 업로드 또는 기본 파일 사용")
uploaded = st.file_uploader("small_cases.csv 업로드", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    default_path = os.path.join(DATA_DIR, EVAL_FILE)
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        st.error(f"기본 파일을 찾을 수 없습니다: {default_path}")
        st.stop()

required_cols = CONTROL_COLS_9 + MONITOR_COLS_28 + [RES_COL, RES_T_COL, BLA_COL, BLA_T_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"CSV에 필요한 컬럼이 없습니다: {missing}")
    st.stop()

df = df.dropna(subset=required_cols).reset_index(drop=True)

st.write("업로드/로드된 데이터:")
st.dataframe(df, use_container_width=True)

st.subheader("2) 실행할 행(row) 선택")
row_idx = st.number_input("row index", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
row = df.iloc[int(row_idx)]

# payload 초기 생성 (CSV row 기반)
payload = {
    "controls": {c: float(row[c]) for c in CONTROL_COLS_9},
    "monitors": {c: float(row[c]) for c in MONITOR_COLS_28},
    "quality": {RES_COL: float(row[RES_COL]), BLA_COL: float(row[BLA_COL])},
    "targets": {RES_T_COL: float(row[RES_T_COL]), BLA_T_COL: float(row[BLA_T_COL])},
    "delta_max": float(delta_max),
}

st.subheader("3) 입력값 확인/수정")
colA, colB = st.columns(2)

with colA:
    st.markdown("### 제어변수(9) - 수정 가능")
    for c in CONTROL_COLS_9:
        payload["controls"][c] = st.number_input(c, value=float(payload["controls"][c]), format="%.6f")

    st.markdown("### 품질/목표 - 수정 가능")
    payload["quality"][RES_COL] = st.number_input("residue (현재)", value=float(payload["quality"][RES_COL]), format="%.6f")
    payload["targets"][RES_T_COL] = st.number_input("residue_target", value=float(payload["targets"][RES_T_COL]), format="%.6f")
    payload["quality"][BLA_COL] = st.number_input("blaine (현재)", value=float(payload["quality"][BLA_COL]), format="%.6f")
    payload["targets"][BLA_T_COL] = st.number_input("blaine_target", value=float(payload["targets"][BLA_T_COL]), format="%.6f")

with colB:
    st.markdown("### 모니터링(28) - 읽기 전용")
    with st.expander("모니터링 변수 보기 (28개)", expanded=False):
        mon_view = pd.DataFrame([payload["monitors"]]).T.reset_index()
        mon_view.columns = ["monitor", "value"]
        st.dataframe(mon_view, use_container_width=True)

payload["delta_max"] = float(delta_max)

# 검증(controls+monitors 모두 실행 차단, ±2% margin)
errors = validate_payload_controls_only(payload, bundle["CONTROL_LIMITS_M"])

st.subheader("4) 실행")
if errors:
    st.error("입력값 오류로 실행할 수 없습니다. 아래 항목을 수정하세요.")
    st.write(errors)
    run_disabled = True
else:
    st.success("입력 검증 통과 (실행 가능)")
    run_disabled = False

if st.button("1-step 추천 실행", disabled=run_disabled):
    result = recommend_1step_from_payload(payload, bundle=bundle)

    st.subheader("✅ 추천 결과 (CONTROL_COLS_9 순서)")
    out_rows = []
    for c in CONTROL_COLS_9:
        out_rows.append({
            "control": c,
            "u_cur": float(payload["controls"][c]),
            "u_new": float(result["u_new"][c]),
            "delta": float(result["delta"][c]),
        })
    out_df = pd.DataFrame(out_rows)
    st.dataframe(out_df, use_container_width=True)

    st.subheader("📈 품질 예측(현재 → 1-step 적용 후)")
    st.json(result["pred"])

    st.subheader("📌 SPEC 판정")
    st.json(result["spec"])

    # ---- CSV 다운로드(요약 1행)
    summary_row = {
        "row_idx": int(row_idx),
        "delta_max": float(result["pred"]["delta_max"]),
        "alpha": float(result["pred"]["alpha"]),
        "residue_cur": float(result["pred"]["residue_cur"]),
        "residue_next": float(result["pred"]["residue_next"]),
        "blaine_cur": float(result["pred"]["blaine_cur"]),
        "blaine_next": float(result["pred"]["blaine_next"]),
        "residue_next_ok": bool(result["spec"]["residue_next_ok"]),
        "blaine_next_ok": bool(result["spec"]["blaine_next_ok"]),
        "both_next_ok": bool(result["spec"]["both_next_ok"]),
    }
    for c in CONTROL_COLS_9:
        summary_row[f"{c}_cur"] = float(payload["controls"][c])
        summary_row[f"{c}_new"] = float(result["u_new"][c])
        summary_row[f"{c}_delta"] = float(result["delta"][c])

    summary_df = pd.DataFrame([summary_row])

    csv_buffer = io.StringIO()
    summary_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    csv_bytes = csv_buffer.getvalue().encode("utf-8-sig")

    st.download_button(
        label="결과 CSV 다운로드 (1행 요약)",
        data=csv_bytes,
        file_name=f"recommend_result_row{int(row_idx)}.csv",
        mime="text/csv"
    )
