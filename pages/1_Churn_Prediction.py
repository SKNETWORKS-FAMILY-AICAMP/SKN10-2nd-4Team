import os
import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ✅ 현재 `pages/1_Churn_Prediction.py` 파일의 위치를 기준으로 루트 디렉토리 찾기
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # `pages/` 상위 폴더

# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # ✅ 마이너스 기호 깨짐 방지

# 📌 모델 경로 탐색 및 자동 리스트업
MODEL_DIR = os.path.join(BASE_DIR, "model")
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

# ✅ Streamlit UI 설정
st.set_page_config(page_title="Churn Prediction", layout="wide")

# st.title("📊 고객 이탈 예측 - 모델별 정확도 비교")
# cluster0_model_lgbm = os.path.join(BASE_DIR, "streamlit/img", "cluster0_model_lgbm.png")
# st.image(cluster0_model_lgbm, caption="클러스터링 최적화 분석 결과", width=200, use_container_width=True)
# cluster3_model_lgbm = os.path.join(BASE_DIR, "streamlit/img", "cluster3_model_lgbm.png")
# st.image(cluster3_model_lgbm, caption="클러스터링 최적화 분석 결과", width=200, use_container_width=True)

# ✅ 이미지 경로 설정
cluster0_model_lgbm_path = os.path.join(BASE_DIR, "streamlit/img", "cluster0_model_rf_comparison.jpg")
cluster3_model_lgbm_path = os.path.join(BASE_DIR, "streamlit/img", "cluster3_model_rf_comparison.jpg")

# ✅ 이미지 크기 조절 함수
def resize_image(image_path, scale=0.5):
    """Pillow를 사용하여 이미지 크기를 줄이는 함수"""
    try:
        image = Image.open(image_path)
        new_size = (int(image.width * scale), int(image.height * scale))  # 비율 조절
        resized_image = image.resize(new_size)
        return resized_image
    except Exception as e:
        st.error(f"❌ 이미지 로드 실패: {e}")
        return None

# ✅ 이미지 로드 및 표시 (PIL을 사용하여 크기 조절)
cluster0_image = resize_image(cluster0_model_lgbm_path)
if cluster0_image:
    st.image(cluster0_image, caption="클러스터 0 - 최적화 분석 결과")

cluster3_image = resize_image(cluster3_model_lgbm_path)
if cluster3_image:
    st.image(cluster3_image, caption="클러스터 3 - 최적화 분석 결과")

st.title("📊 고객 이탈 예측 - 두 모델 정확도 비교")
# 📌 모델 선택 및 로드
st.sidebar.header("🔍 모델 설정")
selected_models = st.sidebar.multiselect("비교할 모델 선택 (최대 2개)", model_files, default=model_files[:2])

# 📌 데이터 불러오기
st.sidebar.header("📂 데이터")
data_file = st.sidebar.selectbox("사용할 데이터 선택", ["Cluster_0_Preprocessed.csv", "Cluster_3_Preprocessed.csv"])

if st.sidebar.button("📊 데이터 불러오기"):
    csv_path = os.path.join(BASE_DIR, "data", data_file)
    try:
        df = pd.read_csv(csv_path)
        st.session_state["data"] = df  # ✅ 데이터 세션에 저장
        st.success(f"✅ 데이터 불러오기 완료! (총 {len(df)} 행)")
    except FileNotFoundError:
        st.error(f"❌ 데이터 파일을 찾을 수 없습니다: {csv_path}")

# ✅ 현재 로드된 데이터 미리보기
if "data" in st.session_state:
    st.write("📊 고객 데이터 미리보기", st.session_state["data"].head())

st.sidebar.header("📈 모델 정확도 비교")

# ✅ 모델 비교 실행
if st.sidebar.button("🚀 모델 정확도 평가 실행"):
    if len(selected_models) != 2:
        st.warning("📌 두 개의 모델을 선택해야 합니다!")
    else:
        models = {}
        for model_name in selected_models:
            model_path = os.path.join(MODEL_DIR, model_name)
            models[model_name] = joblib.load(model_path)

        # ✅ 데이터 로드 확인
        if "data" not in st.session_state:
            st.warning("📌 먼저 데이터를 불러오세요!")
        else:
            df = st.session_state["data"]
            
            if "Exited" not in df.columns:
                st.error("❌ 데이터에 'Exited' 컬럼이 없습니다. 올바른 데이터셋을 사용하세요.")
            else:
                X = df.drop(columns=["Exited"])
                y = df["Exited"]

                results = {}
                
                for model_name, model in models.items():
                    expected_features = model.feature_names_in_

                    # 부족한 컬럼 0으로 채우기
                    for col in expected_features:
                        if col not in X.columns:
                            X[col] = 0  

                    # ✅ 모델 입력에 맞게 feature 정렬
                    X_sorted = X[expected_features]

                    # ✅ 예측 실행
                    y_pred = model.predict(X_sorted)

                    # ✅ 성능 평가
                    acc = accuracy_score(y, y_pred)
                    f1 = f1_score(y, y_pred)
                    precision = precision_score(y, y_pred)
                    recall = recall_score(y, y_pred)

                    results[model_name] = {
                        "Accuracy": acc,
                        "F1-score": f1,
                        "Precision": precision,
                        "Recall": recall,
                    }

                # ✅ 결과 출력
                st.subheader("📊 모델별 정확도 비교")
                for model_name, metrics in results.items():
                    st.write(f"🔹 **{model_name}**")
                    st.write(f"  - Accuracy: {metrics['Accuracy']:.4f}")
                    st.write(f"  - F1-score: {metrics['F1-score']:.4f}")
                    st.write(f"  - Precision: {metrics['Precision']:.4f}")
                    st.write(f"  - Recall: {metrics['Recall']:.4f}")

                # ✅ 그래프 시각화
                fig, ax = plt.subplots(figsize=(8, 5))
                accuracy_values = [results[m]["Accuracy"] for m in selected_models]
                f1_values = [results[m]["F1-score"] for m in selected_models]

                bar_width = 0.3
                index = range(len(selected_models))

                ax.bar(index, accuracy_values, bar_width, label="Accuracy", color='#ff9999')
                ax.bar([i + bar_width for i in index], f1_values, bar_width, label="F1-score", color='#66b3ff')

                ax.set_xticks([i + bar_width / 2 for i in index])
                ax.set_xticklabels(selected_models)
                ax.set_ylabel("Score")
                ax.set_title("모델 성능 비교")
                ax.legend()

                for i, v in enumerate(accuracy_values):
                    ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=10, fontweight='bold')

                for i, v in enumerate(f1_values):
                    ax.text(i + bar_width, v + 0.02, f"{v:.4f}", ha='center', fontsize=10, fontweight='bold')

                st.pyplot(fig)
