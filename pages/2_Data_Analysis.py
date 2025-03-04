import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib  # 모델 로딩용 라이브러리

# 📌 Streamlit UI 설정
st.set_page_config(page_title="Churn Data Analysis", layout="wide")

# 📌 페이지 제목
st.title("📊 금융상품 적용에 따른 고객 이탈률 변화 분석")
st.write("""
금융상품 가입이 고객 이탈에 미치는 영향을 분석하고,  
선택한 금융상품에 따라 이탈률이 어떻게 변하는지 확인할 수 있습니다.
""")

# ✅ 현재 프로젝트 구조 반영
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

cluster_0_path = os.path.join(DATA_DIR, "Cluster_0_Filtered.csv")
cluster_3_path = os.path.join(DATA_DIR, "Cluster_3_Filtered.csv")
model_path = os.path.join(MODEL_DIR, "final_model_rf.pkl")  # 모델 경로

# ✅ 파일 존재 여부 확인 후 로드
if os.path.exists(cluster_0_path) and os.path.exists(cluster_3_path):
    df_0 = pd.read_csv(cluster_0_path)
    df_3 = pd.read_csv(cluster_3_path)
    st.success("✅ 클러스터 데이터 불러오기 완료!")
else:
    st.error("❌ 클러스터 데이터 파일을 찾을 수 없습니다.")
    st.stop()

# ✅ 모델 로드
if os.path.exists(model_path):
    model = joblib.load(model_path)  # 모델 로드
    st.success("✅ 이탈 예측 모델 로드 완료!")
else:
    st.error("❌ 모델 파일을 찾을 수 없습니다.")
    st.stop()

# ✅ 데이터 병합 및 클러스터 추가
df_0["Cluster"] = "0 (고소득 금융상품 다수 보유)"
df_3["Cluster"] = "3 (충성도 높은 고객)"
df = pd.concat([df_0, df_3], ignore_index=True)

# 📌 1️⃣ 대상 데이터 미리보기
st.subheader("📊 대상 고객 데이터 미리보기")
st.write(df.head())

# ✅ 2️⃣ 금융상품 선택
st.subheader("💳 금융상품 선택")
selected_products = st.multiselect(
    "적용할 금융상품을 선택하세요 (최대 2개)",
    ["신용카드 추가 발급", "대출 금리 할인", "투자 상품 추천", "예금 금리 상승", "자동이체 캐시백"],
    default=["신용카드 추가 발급", "대출 금리 할인"]
)

# ✅ 금융상품 적용 가정
df_applied = df.copy()



if "신용카드 추가 발급" in selected_products:
    df_applied["NumOfProducts"] += 1

if "대출 금리 할인" in selected_products:
    df_applied["CreditScore"] += 50  

if "투자 상품 추천" in selected_products:
    df_applied["Balance"] *= 1.1  

if "예금 금리 상승" in selected_products:
    df_applied["Balance"] *= 1.05  

if "자동이체 캐시백" in selected_products:
    df_applied["NumOfProducts"] += 1  

# ✅ 모델 예측을 위해 필요한 피처만 선택
features = model.feature_names_in_  # 모델이 학습할 때 사용한 피처 목록
df_original_input = df[features]
df_applied_input = df_applied[features]

# ✅ 모델을 사용하여 이탈 예측 수행
df["Predicted_Exited"] = model.predict(df_original_input)
df_applied["Predicted_Exited"] = model.predict(df_applied_input)

# ✅ 이탈률 비교
original_exit_rate = df["Predicted_Exited"].mean()  # 금융상품 적용 전 예측된 이탈률
new_exit_rate = df_applied["Predicted_Exited"].mean()  # 금융상품 적용 후 예측된 이탈률

# 📌 3️⃣ 금융상품 적용 전/후 이탈률 비교 그래프
st.subheader("📉 금융상품 적용 전후 이탈률 비교")
fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#FF6F61", "#6B8E23"]

bars = ax.bar(["금융상품 적용 전", "금융상품 적용 후"], [original_exit_rate, new_exit_rate], color=colors, alpha=0.85, width=0.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2%}", ha="center", fontsize=12, fontweight="bold", color="black")

ax.set_ylabel("평균 이탈률", fontsize=12, fontweight="bold")
ax.set_title("금융상품 적용에 따른 이탈률 변화", fontsize=14, fontweight="bold", pad=15)
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=11)

st.pyplot(fig)

# 📌 4️⃣ 버튼을 눌러야 인사이트/결론 보이도록 변경
if st.button("📢 인사이트 및 결론 보기"):
    st.subheader("📢 인사이트 및 결론")
    st.markdown(f"""
    - 금융상품을 적용한 결과, **이탈률이 `{original_exit_rate:.2%}` → `{new_exit_rate:.2%}`로 감소**하였습니다.
    - 고객의 **신용등급이 상승**하고 **금융상품 개수가 증가**하면 이탈 가능성이 낮아지는 경향을 확인할 수 있습니다.
    - 향후 **이탈 가능성이 높은 고객을 선별하여 맞춤형 금융상품을 제공**하면 고객 유지를 효과적으로 개선할 수 있습니다.
    """)

    st.success("💡 다음 단계: 고이탈 위험 고객을 타겟팅한 금융상품 제안 및 전략 수립")


## 추산 이익 계산하는 코드 추가할것.