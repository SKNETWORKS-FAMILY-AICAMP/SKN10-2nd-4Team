import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 📌 Streamlit UI 설정
st.set_page_config(page_title="Financial Product Simulation", layout="wide")

# 📌 페이지 제목
st.title("💰 금융상품 적용 시뮬레이션")
st.write("금융상품을 적용하여 고객 이탈률이 어떻게 변하는지 시뮬레이션합니다.")

# 📌 사용자 입력: 금융상품 추가 시뮬레이션
st.sidebar.header("🔧 금융상품 설정")

num_products = st.sidebar.slider("📈 금융상품 개수 변경", 1, 5, 2)
credit_score = st.sidebar.slider("💳 신용도 점수 조정", 300, 850, 600)

st.write(f"📌 현재 선택된 금융상품 개수: {num_products} 개")
st.write(f"📌 현재 선택된 신용도 점수: {credit_score}")

# 📌 금융상품 적용 버튼
if st.button("🚀 금융상품 적용하기"):
    
    # ✅ 금융상품 선택 박스 표시
    st.subheader("📦 적용할 금융상품 선택")
    product_options = [
        "🔹 예금 상품 (Deposit)",
        "💳 신용카드 (Credit Card)",
        "📈 투자 상품 (Investment)",
        "🏠 주택 대출 (Mortgage)",
        "🚗 자동차 대출 (Auto Loan)"
    ]
    
    selected_products = st.multiselect("적용할 금융상품을 선택하세요:", product_options)

    # ✅ 이탈률 감소 효과 반영
    base_reduction = 0.05  # 기본 금융상품 적용 시 이탈률 감소율
    reduction_factor = len(selected_products) * 0.02  # 금융상품 개수에 따른 추가 감소율
    final_reduction = base_reduction + reduction_factor

    # 적용 후 데이터 시뮬레이션
    np.random.seed(42)
    num_customers = 1000
    original_churn_rate = np.random.uniform(0.1, 0.3, num_customers)  # 기존 이탈률 10~30%
    new_churn_rate = original_churn_rate * (1 - final_reduction)  # 적용된 이탈률 감소 반영

    # 차트 시각화
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(original_churn_rate, bins=30, alpha=0.5, label="📉 금융상품 적용 전")
    ax.hist(new_churn_rate, bins=30, alpha=0.7, label="📈 금융상품 적용 후")
    ax.legend()
    ax.set_title("📊 금융상품 적용 전/후 이탈률 변화")

    st.pyplot(fig)

    # 금융상품 적용 후 예상 피처값
    updated_data = pd.DataFrame({
        "고객 ID": np.arange(1, num_customers+1),
        "이전 이탈률": original_churn_rate,
        "적용 후 이탈률": new_churn_rate,
        "적용 금융상품 개수": len(selected_products)
    })
    st.write("🔍 금융상품 적용 후 변경된 고객 데이터 일부 미리보기")
    st.dataframe(updated_data.sample(5))


# 📌 추산 이익 계산 버튼
if st.button("💰 추산 이익 계산하기"):
    # 가입자 증가율 예측
    num_customers = 1000
    base_conversion_rate = 0.1  # 기본 금융상품 가입률 (10%)
    expected_new_customers = int(num_customers * base_conversion_rate * (1 + len(selected_products) * 0.05))

    # ✅ 1인당 예상 수익 계산 (600만원 = 6,000,000원)
    revenue_per_customer = 6000000  # 한 명당 수익 (600만원)
    estimated_profit = expected_new_customers * revenue_per_customer  # 전체 예상 이익

    # ✅ 결과 표시
    st.subheader("📊 금융상품 가입 예상 고객 수 및 추산 이익")
    result_df = pd.DataFrame({
        "항목": ["예상 신규 가입자 수", "추산 이익 (원)"],
        "값": [expected_new_customers, f"{estimated_profit:,.0f} 원"]
    })
    st.table(result_df)