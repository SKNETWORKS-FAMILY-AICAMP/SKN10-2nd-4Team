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
    st.write("📊 고객 데이터 미리보기", data.head())

# 📌 이탈 예측 실행
st.sidebar.header("예측")
if st.sidebar.button("이탈 예측 실행"):
    if 'model' in locals():
        sample = data.drop(columns=["Exited"]).sample(1)
        prediction = model.predict(sample)
        result = "🚨 이탈" if prediction[0] == 1 else "✅ 잔류"
        st.write("📌 예측 결과:", result)
    else:
        st.warning("📌 먼저 모델을 로드하세요!")