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

# ✅ 2️⃣ 금융상품 선택
st.subheader("💳 금융상품 선택")
selected_products = st.multiselect(
    "적용할 금융상품을 선택하세요 (최대 2개)",
    [
        "이탈 고객 활동 멤버 전환",
        "연이율 3% 복리 금융상품 가입 (2년)",
        "신용카드 신규 가입 + 포인트 지급",
    ],
    # default=["신용카드 신규 가입 + 포인트 지급", "연이율 3% 복리 금융상품 가입 (2년)"]
)

# ✅ 금융상품 적용 가정
df_applied = df.copy()

if "이탈 고객 활동 멤버 전환" in selected_products:
    df_applied.loc[df_applied["IsActiveMember"] == 0, "IsActiveMember"] = 1

if "연이율 3% 복리 금융상품 가입 (2년)" in selected_products:
    df_applied["Balance"] *= 1.0609
    df_applied["Tenure"] += 2

if "신용카드 신규 가입 + 포인트 지급" in selected_products:
    df_applied.loc[df_applied["HasCrCard"] == 0, "Point Earned"] += 40
    df_applied.loc[df_applied["HasCrCard"] == 0, "HasCrCard"] = 1
# ✅ 모델 예측을 위해 필요한 피처만 선택
features = model.feature_names_in_
df_original_input = df[features]
df_applied_input = df_applied[features]


######################################################################################################


# ✅ 이탈 고객 필터링
exited_mem = df[df["Exited"] == 1].copy()

# ✅ 금융상품 적용 전 이탈 확률 계산
prob_original = model.predict_proba(exited_mem[features])[:, 1]

# ✅ 금융상품 적용 후 데이터 변경
if "이탈 고객 활동 멤버 전환" in selected_products:
    exited_mem.loc[exited_mem["IsActiveMember"] == 0, "IsActiveMember"] = 1

if "연이율 3% 복리 금융상품 가입 (2년)" in selected_products:
    exited_mem["Balance"] *= 1.0609
    exited_mem["Tenure"] += 2

if "신용카드 신규 가입 + 포인트 지급" in selected_products:
    exited_mem.loc[exited_mem["HasCrCard"] == 0, "Point Earned"] += 40
    exited_mem.loc[exited_mem["HasCrCard"] == 0, "HasCrCard"] = 1

# ✅ 금융상품 적용 후 이탈 확률 예측
prob_applied = model.predict_proba(exited_mem[features])[:, 1]

# ✅ 이탈률 비교 (평균값)
original_exit_rate = np.mean(prob_original)
new_exit_rate = np.mean(prob_applied)

# ✅ 이탈 확률 감소 고객 수 및 평균 감소량 계산
reduced_exit_count = np.sum(prob_applied < prob_original)
total_exit_count = len(prob_original)
reduced_exit_ratio = reduced_exit_count / total_exit_count * 100
prob_change = np.mean(prob_original - prob_applied)

# ✅ 📊 이탈 확률 변화 그래프 시각화
st.subheader("📉 금융상품 적용 전후 이탈 확률 비교")
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["금융상품 적용 전", "금융상품 적용 후"], 
              [original_exit_rate, new_exit_rate], 
              color=["#FF6F61", "#6B8E23"], alpha=0.85, width=0.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2%}", 
            ha="center", fontsize=12, fontweight="bold", color="black")

ax.set_ylabel("평균 이탈 확률", fontsize=12, fontweight="bold")
ax.set_title("금융상품 적용에 따른 이탈률 변화", fontsize=14, fontweight="bold", pad=15)

st.pyplot(fig)

# ✅ 📌 이탈 확률 감소 데이터 출력
st.write(f"금융상품 적용 전 평균 이탈 확률: **{original_exit_rate:.2%}**")
st.write(f"금융상품 적용 후 평균 이탈 확률: **{new_exit_rate:.2%}**")
st.write(f"이탈 확률 감소 고객 수: **{reduced_exit_count}/{total_exit_count}명**")
st.write(f"이탈 확률 감소 비율: **{reduced_exit_ratio:.2f}%**")
st.write(f"평균 이탈 확률 감소량: **{prob_change:.2%}**")

# ✅ 이탈률 감소 효과에 따른 예상 추가 수익 계산
st.subheader("💰 금융상품 도입 시 예상 이익 분석")

# 📌 한국 주요 은행 연평균 매출 데이터 (조 원 단위)
bank_revenue_range = (30e12, 40e12)  # 30~40조 원
average_customer_range = (15000000, 20000000)  # 고객 수 1500~2000만 명

# ✅ 1인당 연평균 매출(ARPU) 계산
min_arpu = bank_revenue_range[0] / average_customer_range[1]
max_arpu = bank_revenue_range[1] / average_customer_range[0]

# ✅ 실제 은행 이용 고객 수 반영
actual_total_customers = 15000000  # 1500만 명
cluster_customer_ratio = 0.24  # 분석 대상 고객군 비율
scaled_total_customers = actual_total_customers * cluster_customer_ratio

# ✅ 금융상품 적용 전후 예상 이탈자 수
original_exited_count = original_exit_rate * scaled_total_customers
new_exited_count = new_exit_rate * scaled_total_customers
reduced_exited_count = original_exited_count - new_exited_count  # 줄어든 이탈자 수

# ✅ 이탈자 감소 효과에 따른 수익 증가 계산
average_revenue_per_customer = 500000  # 고객 1명당 평균 연매출 (50만 원 가정)
estimated_additional_revenue = reduced_exited_count * average_revenue_per_customer

# ✅ 금액을 읽기 쉽게 변환하는 함수
def format_revenue(amount):
    """금액을 'X억 Y천만 원' 형식으로 변환"""
    if amount >= 1_0000_0000:  
        eok = int(amount // 1_0000_0000)
        chonman = int((amount % 1_0000_0000) // 1_0000_000)
        return f"약 {eok}억 {chonman}천만 원" if chonman > 0 else f"약 {eok}억 원"
    elif amount >= 1_0000_000:
        chonman = int(amount // 1_0000_000)
        return f"약 {chonman}천만 원"
    else:
        return f"약 {amount:,.0f} 원"
# 📌 한국 주요 은행 연매출 데이터 (조 원 단위)
bank_revenue_range = (30e12, 40e12)  # 30~40조 원
average_customer_range = (15000000, 20000000)  # 고객 수 1500~2000만 명

# ✅ 1인당 연평균 매출(ARPU) 자동 계산
min_arpu = bank_revenue_range[0] / average_customer_range[1]  # 최소 추정값 (150만 원)
max_arpu = bank_revenue_range[1] / average_customer_range[0]  # 최대 추정값 (266만 원)

# ✅ 현실적인 중간값 사용 (180만 원 적용)
default_arpu = (min_arpu + max_arpu) / 2  # 180만 원

# 📌 사용자 입력을 통한 ARPU 조정 가능 (Streamlit 적용)
st.sidebar.subheader("📊 평균 고객 연매출 (ARPU) 설정")
selected_arpu = st.sidebar.slider(
    "1인당 연평균 매출 (ARPU) 설정",
    min_value=int(min_arpu),
    max_value=int(max_arpu),
    value=int(default_arpu),  # 기본값: 180만 원
    step=100000  # 10만 원 단위 조정 가능
)

# ✅ ARPU 값 적용
average_revenue_per_customer = selected_arpu

# ✅ 예상 추가 수익 계산
estimated_additional_revenue = reduced_exited_count * average_revenue_per_customer

# ✅ 예상 추가 수익 출력
st.subheader("💰 금융상품 도입 시 예상 이익 분석")
st.write(f"**줄어든 이탈자 수:** {reduced_exited_count:,.0f} 명")
st.write(f"**추정 1인당 연매출 (ARPU):** {average_revenue_per_customer:,.0f} 원")
st.write(f"**추가 예상 이익:** {format_revenue(estimated_additional_revenue)}")


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
