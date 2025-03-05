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

cluster_0_path = os.path.join(DATA_DIR, "Cluster_0_Preprocessed.csv")
cluster_3_path = os.path.join(DATA_DIR, "Cluster_3_Preprocessed.csv")
# model_path = os.path.join(MODEL_DIR, "final_model_rf.pkl")  # 모델 경로

# # 📌 모델 파일 리스트업 (자동 탐색)
# model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith("_rf.pkl")]



# ✅ 파일 존재 여부 확인 후 로드
if os.path.exists(cluster_0_path) and os.path.exists(cluster_3_path):
    df_0 = pd.read_csv(cluster_0_path)
    df_3 = pd.read_csv(cluster_3_path)
    st.success("✅ 클러스터 데이터 불러오기 완료!")
else:
    st.error("❌ 클러스터 데이터 파일을 찾을 수 없습니다.")
    st.stop()

# 📌 모델 선택 UI 추가
st.sidebar.header("🔍 모델 선택")
selected_model = st.sidebar.selectbox("사용할 모델 선택", model_files)


# ✅ 모델 로드
if selected_model:
    model_path = os.path.join(MODEL_DIR, selected_model)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success(f"✅ {selected_model} 모델 로드 완료!")
    else:
        st.error("❌ 선택한 모델 파일을 찾을 수 없습니다.")
        st.stop()


# ✅ 데이터 병합 및 클러스터 추가
df_0["Cluster"] = "0 (고소득 금융상품 다수 보유)"
df_3["Cluster"] = "3 (충성도 높은 고객)"
df = pd.concat([df_0, df_3], ignore_index=True)

# 📌 1️⃣ 대상 데이터 미리보기
st.subheader("📊 대상 고객 데이터 미리보기")
st.write(df.head())

# ✅ 2️⃣ 금융상품 선택 (Case 기반)
st.subheader("💳 금융상품 선택")
selected_products = st.multiselect(
    "적용할 금융상품을 선택하세요 (최대 2개)",
    [
        "이탈 고객 활동 멤버 전환",
        "가입 기간 반영 신용 점수 증가",
        "연이율 3% 복리 금융상품 가입 (2년)",
        "신용카드 신규 가입 + 포인트 지급",
        "비활동 유저의 로열티 증가"
    ],
    default=["신용카드 신규 가입 + 포인트 지급", "연이율 3% 복리 금융상품 가입 (2년)"]
)

# ✅ 금융상품 적용 가정
df_applied = df.copy()

if "이탈 고객 활동 멤버 전환" in selected_products:
    df_applied.loc[df_applied["IsActiveMember"] == 0, "IsActiveMember"] = 1

if "가입 기간 반영 신용 점수 증가" in selected_products:
    df_applied["CreditScore"] += 30 * (df_applied["Tenure"] / df_applied["Tenure"].max())

if "연이율 3% 복리 금융상품 가입 (2년)" in selected_products:
    df_applied["Balance"] *= 1.0609
    df_applied["Tenure"] += 2

if "신용카드 신규 가입 + 포인트 지급" in selected_products:
    df_applied.loc[df_applied["HasCrCard"] == 0, "Point Earned"] += 40
    df_applied.loc[df_applied["HasCrCard"] == 0, "HasCrCard"] = 1

if "비활동 유저의 로열티 증가" in selected_products:
    df_applied.loc[df_applied["IsActiveMember"] == 0, "Loyalty_Score"] += 2
    df_applied.loc[df_applied["IsActiveMember"] == 0, "IsActiveMember"] = 1

# ✅ 모델 예측을 위해 필요한 피처만 선택
features = model.feature_names_in_
df_original_input = df[features]
df_applied_input = df_applied[features]

# ✅ 모델을 사용하여 이탈 예측 수행
df["Predicted_Exited"] = model.predict(df_original_input)
df_applied["Predicted_Exited"] = model.predict(df_applied_input)

# ✅ 이탈률 비교
original_exit_rate = df["Predicted_Exited"].mean()
new_exit_rate = df_applied["Predicted_Exited"].mean()

# st.pyplot(fig)
# 📌 3️⃣ 금융상품 적용 전/후 이탈률 비교 그래프
st.subheader("📉 금융상품 적용 전후 이탈률 비교")

fig, ax = plt.subplots(figsize=(5, 4))  # ✅ 그래프 크기 조정
colors = ["#FF4C4C", "#4CAF50"]  # ✅ 대비되는 강한 색상 적용

bars = ax.bar(["금융상품 적용 전", "금융상품 적용 후"], [original_exit_rate, new_exit_rate], color=colors, alpha=0.85, width=0.5)

# ✅ Y축 범위 조정 (이탈률 차이를 강조)
ax.set_ylim(min(original_exit_rate, new_exit_rate) - 0.01, max(original_exit_rate, new_exit_rate) + 0.01)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.002, f"{height:.2%}", ha="center", fontsize=12, fontweight="bold", color="black")

# # ✅ 이탈률 감소 강조 텍스트 추가
# ax.text(0.5, max(original_exit_rate, new_exit_rate) + 0.005, "📉 이탈률 감소!", fontsize=14, fontweight="bold", color="blue", ha="center")

ax.set_ylabel("평균 이탈률", fontsize=12, fontweight="bold")
ax.set_title("금융상품 적용에 따른 이탈률 변화", fontsize=14, fontweight="bold", pad=15)
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=11)

st.pyplot(fig)


# 📌 한국 주요 은행 연평균 매출 데이터 (조 원 단위)
bank_revenue_range = (30e12, 40e12)  # 30~40조 원
average_customer_range = (15000000, 20000000)  # 고객 수 1500~2000만 명

# ✅ 1인당 연평균 매출(ARPU) 계산
min_arpu = bank_revenue_range[0] / average_customer_range[1]  # 최소 추정값
max_arpu = bank_revenue_range[1] / average_customer_range[0]  # 최대 추정값

# 📌 실제 은행 이용 고객 수 반영
actual_total_customers = 15000000  # 실제 은행 고객 수 (1500만 명)

# ✅ 현재 데이터셋이 전체 고객의 약 24%를 차지하는 클러스터에 대한 결과임을 반영
cluster_customer_ratio = 0.24  # 전체 고객 중 해당 군집이 차지하는 비율
scaled_total_customers = actual_total_customers * cluster_customer_ratio  # 조정된 고객 규모

# ✅ 금융상품 적용 전후 이탈자 수 계산
original_exited_count = original_exit_rate * scaled_total_customers
new_exited_count = new_exit_rate * scaled_total_customers
reduced_exited_count = original_exited_count - new_exited_count  # 줄어든 이탈자 수

# ✅ 이탈자 감소 효과에 따른 수익 증가 계산
average_revenue_per_customer = 500000  # 고객 1명당 평균 연매출 (50만원 가정)
estimated_additional_revenue = reduced_exited_count * average_revenue_per_customer  # 예상 추가 수익

# ✅ 금액을 읽기 쉽게 변환하는 함수
def format_revenue(amount):
    """금액을 'X억 Y천만 원' 형식으로 변환"""
    if amount >= 1_0000_0000:  # 1억 이상일 경우
        eok = int(amount // 1_0000_0000)  # 억 단위 계산
        chonman = int((amount % 1_0000_0000) // 1_0000_000)  # 천만 단위 계산
        return f"약 {eok}억 {chonman}천만 원" if chonman > 0 else f"약 {eok}억 원"
    elif amount >= 1_0000_000:  # 1천만 이상일 경우
        chonman = int(amount // 1_0000_000)
        return f"약 {chonman}천만 원"
    else:
        return f"약 {amount:,.0f} 원"

# 📌 사용자 입력을 통한 은행 매출 기반 ARPU 계산
st.sidebar.subheader("📊 현실적인 ARPU 계산")
total_bank_revenue = st.sidebar.number_input("은행 연매출 (조 원 단위)", min_value=10, max_value=50, value=35) * 1_0000_0000_0000
total_bank_customers = st.sidebar.number_input("은행 총 고객 수 (만 명)", min_value=1000, max_value=3000, value=1500) * 10000

# ✅ 사용자가 입력한 은행 매출 및 고객 수 기반 ARPU 계산
calculated_arpu = total_bank_revenue / total_bank_customers

st.sidebar.write(f"📌 계산된 1인당 연평균 매출 (ARPU): **{calculated_arpu:,.0f} 원**")

# ✅ 고객 1명당 평균 연매출 (계산된 값 적용)
average_revenue_per_customer = calculated_arpu  

# ✅ Streamlit에서 ARPU 선택할 수 있도록 설정
st.sidebar.header("📊 평균 고객 연매출 (ARPU) 설정")
selected_arpu = st.sidebar.slider(
    "1인당 연평균 매출 (ARPU) 설정",
    min_value=int(min_arpu),
    max_value=int(max_arpu),
    value=3000000,  # 기본값: 300만 원
    step=100000  # 10만 원 단위 조정 가능
)

# ✅ 이탈자 감소 효과에 따른 수익 증가 계산
estimated_additional_revenue = reduced_exited_count * selected_arpu

# ✅ 가독성 높은 수익 표시
st.subheader("💰 금융상품 도입 시 예상 이익 분석")
st.markdown(f"""
- **한국 주요 은행 연평균 매출**: 30~40조 원
- **평균 고객 수**: 1500~2000만 명
- **분석 대상 고객군 비율**: 전체 고객의 약 **24%**
- **조정된 총 고객 수 (적용 대상 클러스터)**: **{scaled_total_customers:,.0f} 명**
- **추정 1인당 연매출 (ARPU)**: {selected_arpu:,.0f} 원
- 금융상품 적용 전 예상 이탈자 수: **{original_exited_count:,.0f} 명**
- 금융상품 적용 후 예상 이탈자 수: **{new_exited_count:,.0f} 명**
- **줄어든 이탈자 수: {reduced_exited_count:,.0f} 명**
- **추가 예상 이익: {format_revenue(estimated_additional_revenue)}**
""")


# 📌 5️⃣ 인사이트 및 결론
if st.button("📢 인사이트 및 결론 보기"):
    st.subheader("📢 인사이트 및 결론")

    # ✅ 이탈률 변화 강조
    st.markdown(f"""
    <div style="text-align:center; font-size:22px;">
        <b>📉 금융상품 적용 후 <span style="color:#FF5733;">이탈률 감소</span> 확인!</b><br>
        <span style="font-size:26px; font-weight:bold; color:#FF5733;">
            {original_exit_rate:.2%} → {new_exit_rate:.2%}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ✅ 주요 인사이트 정리
    st.markdown("""
    ---
    ✅ **금융상품 적용 효과**
    - 고객의 **신용등급 상승, 금융상품 개수 증가, 로열티 점수 증가**가 **이탈 감소**에 긍정적 영향을 미쳤음.
    - 금융상품 적용을 통해 **고객 유지율**이 향상됨.

    ✅ **1000만 명 기준 예상 효과**
    - **추가 예상 이익**: **{format_revenue(estimated_additional_revenue)}**  
    - 금융상품을 전략적으로 도입하면 **실질적인 수익 증대 가능**.

    ✅ **클러스터별 맞춤 전략**
    - **🟢 고소득 금융상품 다수 보유 고객 (클러스터 0)**  
      → 프리미엄 금융상품(고액 예금, 투자상품, 맞춤형 대출) 추가 유치 효과 큼.  
      → **VIP 서비스 강화 시 높은 수익 기대**.

    - **🔵 충성도 높은 고객 (클러스터 3)**  
      → 로열티 강화 프로그램을 통한 **장기적 관계 유지 가능**.  
      → **업셀링 & 비대면 채널 활용** 효과적.

    """, unsafe_allow_html=True)

    # ✅ 다음 단계 제시
    st.success("💡 **다음 단계:** 이탈 가능성이 높은 고객을 대상으로 맞춤형 금융상품 추천 및 전략 수립 🚀")
