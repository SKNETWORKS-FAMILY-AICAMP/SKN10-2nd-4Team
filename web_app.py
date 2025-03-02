# import os
# import streamlit as st
# import pandas as pd
# import joblib
# import pymysql

# # 📌 MySQL 연결 함수
# def get_db_connection():
#     return pymysql.connect(
#         host="localhost",
#         user="CCC",
#         password="CCC1234",
#         database="bank_customer",
#         cursorclass=pymysql.cursors.DictCursor
#     )

# # 📌 모델 로드 함수
# @st.cache_resource
# def load_model(model_type="rf"):
#     return joblib.load(f"model/final_model_{model_type}.pkl")

# # 📌 데이터 불러오기 함수
# def load_data():
#     conn = get_db_connection()
#     query = "SELECT * FROM customers"
#     df = pd.read_sql(query, conn)
#     conn.close()
#     return df

# # 📌 페이지 설정
# st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# # 📌 사이드바 페이지 네비게이션 (Streamlit 기본 기능 활용)
# st.sidebar.title("🔗 페이지 이동")
# page = st.sidebar.radio("이동할 페이지를 선택하세요:", ["🏠 홈", "📊 이탈 예측", "💰 금융상품 적용"])

# # 📌 선택된 페이지에 따라 이동
# if page == "🏠 홈":

#     # 📌 메인 페이지 내용
#     st.title("💡 고객 이탈 예측 시스템")
#     st.write("""
#     본 시스템은 **고객 데이터를 분석하여 이탈 가능성을 예측**하고,  
#     금융상품을 활용하여 이탈률을 줄이는 시뮬레이션을 제공합니다.
#     """)

#     st.write("### 📌 프로젝트 개요")
#     st.image("streamlit/img/main_page.png", caption="고객 이탈 예측 프로젝트 흐름도")

#     st.title("💡 고객 이탈 예측 시스템")
#     st.write("""
#     본 시스템은 **고객 데이터를 분석하여 이탈 가능성을 예측**하고,  
#     금융상품을 활용하여 이탈률을 줄이는 시뮬레이션을 제공합니다.

#     **📌 프로젝트 개요**  
#     SKN Bank의 CEO는 고객 이탈률을 줄이기 위한 방안을 요구하고 있습니다.  
#     우리는 **고객 데이터를 분석**하여 이탈 패턴을 찾고, **효율적인 전략**을 적용할 것입니다.
#     """)

#     # 📌 CEO의 요구사항 정리
#     st.subheader("👨‍💼 CEO의 요구 사항")
#     st.markdown("""
#     - 고객 이탈률을 분석하고 줄일 방법을 찾아라.
#     - **고객 데이터를 군집화하여 유형을 분류하라.**
#     - **엘보 기법과 실루엣 분석을 활용하여 최적 군집을 찾으라.**
#     - **6개 클러스터를 만든 후, 이탈률 감소에 효과적인 군집을 선택하라.**
#     """)

#     # 📌 분석 과정 설명
#     st.header("📊 분석 과정")

#     st.subheader("1️⃣ 이탈률을 어떻게 분석할 것인가?")
#     st.write("""
#     - **고객 데이터를 군집화하여 유형 분류**
#     - **고객 특성을 바탕으로 이탈 가능성을 예측**
#     """)

#     st.subheader("2️⃣ 클러스터링 방식 선택")
#     st.write("""
#     - 초기에는 4개 군집을 고려했으나, **세부 유형 분류가 어렵다고 판단**
#     - **엘보 기법(Elbow Method)과 실루엣 분석(Silhouette Analysis)을 적용하여 최적 군집 수 결정**
#     - 최종적으로 **6개 클러스터로 분류**
#     """)

#     st.subheader("3️⃣ 타겟 군집 선택")
#     st.write("""
#     - 모든 고객의 이탈률을 줄이는 것은 어렵기 때문에 **가장 효과적인 군집을 선정**
#     - 분석 결과, **클러스터 0과 클러스터 3이 가장 유의미한 개선 가능성을 보임**
#     - 따라서 **이 두 군집을 대상으로 금융상품 개선 전략을 적용**
#     """)

#     # 📌 분석 흐름도 이미지 추가 (예제 이미지)
#     st.subheader("🛠 분석 흐름도")
#     # st.image("flowchart.png", caption="고객 이탈 예측 및 감소 분석 흐름")  # 흐름도 이미지 추가

#     # 📌 다음 단계 안내
#     st.header("🚀 다음 단계")
#     st.write("""
#     1. **이탈 예측 모델 학습 → '📊 Churn Prediction' 페이지**  
#     2. **금융상품 적용 시뮬레이션 → '💰 Financial Product Simulation' 페이지**  
#     """)

#     st.success("왼쪽 **사이드바**에서 이탈 예측 또는 금융상품 적용 페이지로 이동하세요.")
#     # 📌 선택된 페이지에 따라 이동


# # elif page == "📊 이탈 예측":
# #     st.switch_page("pages.1_Churn_Prediction")  # ✅ Streamlit의 내장 기능 사용

# # elif page == "💰 금융상품 적용":
# #     st.switch_page("pages.2_Financial_Product")  # ✅ Streamlit의 내장 기능 사용



import os
import streamlit as st
import pandas as pd
import joblib
import pymysql

# 📌 MySQL 연결 함수
def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="CCC",
        password="CCC1234",
        database="bank_customer",
        cursorclass=pymysql.cursors.DictCursor
    )

# 📌 모델 로드 함수
@st.cache_resource
def load_model(model_type="rf"):
    return joblib.load(f"model/final_model_{model_type}.pkl")

# 📌 데이터 불러오기 함수
def load_data():
    conn = get_db_connection()
    query = "SELECT * FROM customers"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 📌 Streamlit 페이지 설정
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# 📌 사이드바 메뉴 (Streamlit 기본 다중 페이지 지원)
st.sidebar.title("🔗 페이지 이동")

# ✅ Streamlit은 `pages/` 폴더를 자동으로 감지하여 페이지 이동 버튼을 생성하므로,
#    추가적인 코드 없이 자동으로 페이지가 인식됩니다.

# 📌 메인 페이지 내용
st.title("💡 고객 이탈 예측 시스템")
st.write("""
본 시스템은 **고객 데이터를 분석하여 이탈 가능성을 예측**하고,  
금융상품을 활용하여 이탈률을 줄이는 시뮬레이션을 제공합니다.
""")

st.write("### 📌 프로젝트 개요")
st.image("streamlit/img/main_page.png", caption="고객 이탈 예측 프로젝트 흐름도")

st.write("""
**📌 프로젝트 개요**  
SKN Bank의 CEO는 고객 이탈률을 줄이기 위한 방안을 요구하고 있습니다.  
우리는 **고객 데이터를 분석**하여 이탈 패턴을 찾고, **효율적인 전략**을 적용할 것입니다.
""")

# 📌 CEO의 요구사항 정리
st.subheader("👨‍💼 CEO의 요구 사항")
st.markdown("""
- 고객 이탈률을 분석하고 줄일 방법을 찾아라.
- **고객 데이터를 군집화하여 유형을 분류하라.**
- **엘보 기법과 실루엣 분석을 활용하여 최적 군집을 찾으라.**
- **6개 클러스터를 만든 후, 이탈률 감소에 효과적인 군집을 선택하라.**
""")

# 📌 분석 과정 설명
st.header("📊 분석 과정")

st.subheader("1️⃣ 이탈률을 어떻게 분석할 것인가?")
st.write("""
- **고객 데이터를 군집화하여 유형 분류**
- **고객 특성을 바탕으로 이탈 가능성을 예측**
""")

st.subheader("2️⃣ 클러스터링 방식 선택")
st.write("""
- 초기에는 4개 군집을 고려했으나, **세부 유형 분류가 어렵다고 판단**
- **엘보 기법(Elbow Method)과 실루엣 분석(Silhouette Analysis)을 적용하여 최적 군집 수 결정**
- 최종적으로 **6개 클러스터로 분류**
""")

st.subheader("3️⃣ 타겟 군집 선택")
st.write("""
- 모든 고객의 이탈률을 줄이는 것은 어렵기 때문에 **가장 효과적인 군집을 선정**
- 분석 결과, **클러스터 0과 클러스터 3이 가장 유의미한 개선 가능성을 보임**
- 따라서 **이 두 군집을 대상으로 금융상품 개선 전략을 적용**
""")

# 📌 분석 흐름도 이미지 추가 (예제 이미지)
st.subheader("🛠 분석 흐름도")
# st.image("flowchart.png", caption="고객 이탈 예측 및 감소 분석 흐름")  # 흐름도 이미지 추가

# 📌 다음 단계 안내
st.header("🚀 다음 단계")
st.write("""
1. **이탈 예측 모델 학습 → '📊 Churn Prediction' 페이지**  
2. **금융상품 적용 시뮬레이션 → '💰 Financial Product Simulation' 페이지**  
""")

st.success("왼쪽 **사이드바**에서 이탈 예측 또는 금융상품 적용 페이지로 이동하세요.")
