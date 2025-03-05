import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ✅ 현재 `pages/1_Churn_Prediction.py` 파일의 위치를 기준으로 루트 디렉토리 찾기
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # `pages/`의 상위 폴더를 기준으로 설정

# ✅ 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic, Linux: NanumGothic)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # ✅ 마이너스 기호 깨짐 방지

# 📌 모델 로드 함수
def load_model(model_type="rf"):
    model_path = os.path.join(BASE_DIR, "model", f"final_model_{model_type}.pkl")
    model = joblib.load(model_path)
    st.session_state["model"] = model  # ✅ 모델을 세션 상태에 저장
    return model

# 📌 데이터 불러오기 함수
def load_data():
    csv_path = os.path.join(BASE_DIR, "data", "Customer-Churn-Records.csv")  # ✅ 올바른 경로 반영
    try:
        df = pd.read_csv(csv_path)
        st.session_state["data"] = df  # ✅ 데이터프레임을 세션 상태에 저장
        return df
    except FileNotFoundError:
        st.error(f"❌ 데이터 파일을 찾을 수 없습니다: {csv_path}")
        return None

# ✅ Streamlit 세션 상태 초기화
if "model" not in st.session_state:
    st.session_state["model"] = None
if "data" not in st.session_state:
    st.session_state["data"] = None

# 📌 Streamlit UI 설정
st.set_page_config(page_title="Churn Prediction", layout="wide")

# 📌 페이지 제목
st.title("📊 고객 이탈 예측")

# 📌 모델 선택 및 로드
st.sidebar.header("모델 설정")
model_type = st.sidebar.selectbox("사용할 모델", ["rf", "xgb", "lgbm"])

if st.sidebar.button("모델 로드"):
    model = load_model(model_type)
    st.success(f"✅ {model_type.upper()} 모델 로드 완료!")

# 📌 데이터 불러오기
st.sidebar.header("데이터")
if st.sidebar.button("데이터 불러오기"):
    data = load_data()
    if data is not None:
        st.write("📊 고객 데이터 미리보기", data.head())

# 📌 이탈 예측 실행
st.sidebar.header("예측")
if st.sidebar.button("이탈 예측 실행"):
    # ✅ 세션 상태에서 모델과 데이터 가져오기
    model = st.session_state.get("model", None)
    data = st.session_state.get("data", None)

    if model is not None and data is not None:
        try:
            # ✅ 모델이 학습할 때 사용한 feature 목록 가져오기
            expected_features = model.feature_names_in_

            # ✅ 예측 데이터 준비 (One-Hot Encoding 적용)
            sample = data.drop(columns=["Exited"]).sample(1)
            sample_encoded = pd.get_dummies(sample)

            # ✅ 부족한 컬럼이 있으면 0으로 채움
            for col in expected_features:
                if col not in sample_encoded.columns:
                    sample_encoded[col] = 0  # ✅ 모델 학습 시 존재했던 컬럼 추가

            # ✅ 예측할 데이터 컬럼 순서 맞추기
            sample_encoded = sample_encoded[expected_features]

            # ✅ 모델 예측 실행
            prediction = model.predict(sample_encoded)
            result = "🚨 이탈" if prediction[0] == 1 else "✅ 잔류"
            st.write("📌 예측 결과:", result)

        except KeyError:
            st.error("❌ 데이터에 'Exited' 컬럼이 없습니다. 올바른 데이터셋을 사용하세요.")
        except ValueError as e:
            st.error(f"❌ 예측 실행 중 오류 발생: {e}")
    else:
        st.warning("📌 먼저 모델과 데이터를 로드하세요!")
