import streamlit as st

# 📌 Streamlit UI 설정
st.set_page_config(page_title="Financial Product Simulation", layout="wide")

# 📌 페이지 제목
st.title("💰 금융상품 적용 시뮬레이션")
st.write("금융상품을 적용하여 고객 이탈률이 어떻게 변하는지 시뮬레이션합니다.")

# 📌 사용자 입력: 금융상품 추가 시뮬레이션
st.sidebar.header("🔧 금융상품 설정")

num_products = st.sidebar.slider("📈 금융상품 개수 변경", 1, 5, 2)
credit_score = st.sidebar.slider("💳 신용도 점수 조정", 300, 850, 600)

st.write(f"현재 선택된 금융상품 개수: {num_products} 개")
st.write(f"현재 선택된 신용도 점수: {credit_score}")

# 📌 금융상품 적용 후 이탈률 변화 시뮬레이션
st.subheader("📊 금융상품 적용 후 예상 이탈률 변화")
st.write("🔍 여기에 금융상품 적용 후 예측 모델을 추가하세요!")