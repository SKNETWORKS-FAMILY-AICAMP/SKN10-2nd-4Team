import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ✅ 현재 `pages/1_Churn_Prediction.py` 파일의 위치를 기준으로 루트 디렉토리 찾기
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # `pages/` 상위 폴더

# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # ✅ 마이너스 기호 깨짐 방지

# 📌 모델 경로 탐색 및 자동 리스트업
MODEL_DIR = os.path.join(BASE_DIR, "model")
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

# ✅ Streamlit 세션 상태 초기화
if "model" not in st.session_state:
    st.session_state["model"] = None
if "data" not in st.session_state:
    st.session_state["data"] = None
if "loaded_model_name" not in st.session_state:
    st.session_state["loaded_model_name"] = None

# 📌 모델 로드 함수
def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    model = joblib.load(model_path)
    st.session_state["model"] = model
    st.session_state["loaded_model_name"] = model_name  # ✅ 현재 로드된 모델 이름 저장
    st.success(f"✅ {model_name} 로드 완료!")

# 📌 데이터 불러오기 함수
def load_data():
    csv_path = os.path.join(BASE_DIR, "data", "Customer-Churn-Records.csv")  # ✅ 데이터 경로
    try:
        df = pd.read_csv(csv_path)
        st.session_state["data"] = df  # ✅ 데이터 세션에 저장
        st.success(f"✅ 데이터 불러오기 완료! (총 {len(df)} 행)")
    except FileNotFoundError:
        st.error(f"❌ 데이터 파일을 찾을 수 없습니다: {csv_path}")

# ✅ Streamlit UI 설정
st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📊 고객 이탈 예측")


# 📌 분석 흐름도 버튼
if st.button("🛠 EDA"):
    st.subheader("🛠 EDA")
    cluster_image_path = os.path.join(BASE_DIR, "streamlit/img", "Targets.jpg")
    st.image(cluster_image_path, caption="클러스터0, 3", use_container_width=True)   
    st.write("""
    - 클러스터 0, 3 데이터셋 고정 > 이탈률 예측 모델 개발 착수
    """)
    cluster_image_path = os.path.join(BASE_DIR, "streamlit/img", "FeatureEngineeringCode.jpg")
    st.image(cluster_image_path, caption="클러스터 최적화 코드", use_container_width=True)   
    cluster_image_path = os.path.join(BASE_DIR, "streamlit/img", "FeatureEngineering.jpg")
    st.image(cluster_image_path, caption="클러스터 최적화 특징 추가", use_container_width=True)    


# 📌 모델 선택 및 로드
st.sidebar.header("🔍 모델 설정")
selected_model = st.sidebar.selectbox("사용할 모델 선택", model_files, index=0)

if st.sidebar.button("🔄 모델 로드"):
    load_model(selected_model)

# ✅ 현재 로드된 모델 표시
if st.session_state["loaded_model_name"]:s
    st.sidebar.success(f"✔ 현재 로드된 모델: {st.session_state['loaded_model_name']}")

# 📌 데이터 불러오기
st.sidebar.header("📂 데이터")
if st.sidebar.button("📊 데이터 불러오기"):
    load_data()

# ✅ 데이터 미리보기 유지
if st.session_state["data"] is not None:
    st.write("📊 고객 데이터 미리보기", st.session_state["data"].head())

# 📌 이탈 예측 실행
st.sidebar.header("📈 예측 실행")
if st.sidebar.button("🚀 이탈 예측 실행"):
    model = st.session_state.get("model", None)
    data = st.session_state.get("data", None)

    if model is not None and data is not None:
        try:
            expected_features = model.feature_names_in_

            # ✅ 예측 데이터 준비
            sample = data.drop(columns=["Exited"]).sample(1)
            sample_encoded = pd.get_dummies(sample)

            # ✅ 부족한 컬럼 채우기
            for col in expected_features:
                if col not in sample_encoded.columns:
                    sample_encoded[col] = 0  # ✅ 0으로 채워줌

            # ✅ 예측 실행
            sample_encoded = sample_encoded[expected_features]
            prediction = model.predict(sample_encoded)
            result = "🚨 이탈" if prediction[0] == 1 else "✅ 잔류"
            st.write("📌 예측 결과:", result)

        except KeyError:
            st.error("❌ 데이터에 'Exited' 컬럼이 없습니다. 올바른 데이터셋을 사용하세요.")
        except ValueError as e:
            st.error(f"❌ 예측 실행 중 오류 발생: {e}")
    else:
        st.warning("📌 먼저 모델과 데이터를 로드하세요!")


