# import os
# import streamlit as st
# import pandas as pd
# import joblib

# # ✅ 프로젝트 루트 디렉토리 설정
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # 📌 모델 로드 함수
# @st.cache_resource
# def load_model(model_type="rf"):
#     return joblib.load(f"model/final_model_{model_type}.pkl")

# # 📌 데이터 불러오기 함수 (CSV 활용)
# @st.cache_data
# def load_data():
#     csv_path = os.path.join(BASE_DIR, "data", "clustered_Customer_Churn.csv")  # CSV 파일 경로
#     df = pd.read_csv(csv_path)
#     return df

# # 📌 Streamlit 페이지 설정
# st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# # # 📌 사이드바 메뉴 (Streamlit 기본 다중 페이지 지원)
# # st.sidebar.title("🔗 페이지 이동")

# # 📌 메인 페이지 내용
# st.title("💡 고객 이탈 예측 시스템")
# st.write("""
# 본 시스템은 **고객 데이터를 분석하여 이탈 가능성을 예측**하고,  
# 금융상품을 활용하여 이탈률을 줄이는 시뮬레이션을 제공합니다.
# """)

# st.write("### 📌 프로젝트 개요")
# st.image("streamlit/img/main_page.jpg", caption="고객 이탈 예측 프로젝트 흐름도")

# st.write("""
# **📌 프로젝트 개요**  
# SKN Bank의 CEO는 고객 이탈률을 줄이기 위한 방안을 요구하고 있습니다.  
# 우리는 SKN Bank사의 데이터 사이언티스트 딤으로 **고객 데이터를 분석**하여 이탈 패턴을 찾고, **효율적인 전략**을 적용할 것입니다.
# """)

# # 📌 CEO 요구사항 버튼
# if st.button("📌 CEO의 요구사항 보기"):
#     st.subheader("👨‍💼 CEO의 요구 사항")
#     st.markdown("""
#     **고객 이탈률을 분석하고 줄일 방법을 찾아라.**
#     """)

# # 📌 분석 과정 버튼
# if st.button("📊 분석 과정 보기"):
#     st.header("📊 분석 과정")

#     st.subheader("1️⃣ 이탈률을 어떻게 분석할 것인가?")
#     st.write("""
#     - **고객 데이터를 군집화하여 유형 분류**
#     - **군집 특성을 바탕으로 이탈 가능성을 예측**해보는건 어떨까?
#     """)

#     st.subheader("2️⃣ 클러스터링 방식 선택")
#     st.write("""
#     - **엘보 기법(Elbow Method)과 실루엣 분석(Silhouette Analysis)을 적용하여 최적 군집 4개**
#     """)
#     # 📌 클러스터링 최적화 이미지 추가
#     cluster_image_path = os.path.join(BASE_DIR, "streamlit/img", "OptimizeCluster.jpg")
#     st.image(cluster_image_path, caption="클러스터링 최적화 분석 결과", use_container_width=True)
#     st.write("""
#     - but 4개 군집을 고려했으나, **세부 유형 분류가 어렵다고 판단**
#     - 최종적으로 **6개 클러스터로 분류**
#     """)
#     # 📌 클러스터 피쳐 이미지 추가
#     ClusterFeature_image_path = os.path.join(BASE_DIR, "streamlit/img", "ClusterFeature.jpg")
#     st.image(ClusterFeature_image_path, caption="클러스터 특징", use_container_width=True)


# if st.button("🛠 군집 분석"):
#     st.subheader("타겟 군집 선택")
#     cluster_image_path = os.path.join(BASE_DIR, "streamlit/img", "ClusterStrategy.jpg")
#     st.image(cluster_image_path, caption="클러스터 전략략", use_container_width=True)    
#     st.write("""
#     - 모든 고객의 이탈률을 줄이는 것은 **비용과 시간** 측면에서 비효율적, 금융상품 제공시 **가장 효과적인 군집**을 선정.
#     - **클러스터 1번과 2번**: 은행의 핵심 상품 전략과 연관성이 낮으며, 이탈 방지를 위한 비용이 상대적으로 클 가능성이 높음.
#     - **클러스터 4번(고령층)**: 금융활동이 정체되었을 가능성이 커 적극적인 마케팅 효과가 제한적임.
#     - **클러스터 5번(젊은 고객층)**은 **장기적인 고객 유치 전략**이 필요하지만, 단기간(1~2년) 내 성과를 내기 어려움
#     - **클러스터 0번과 3번**: **이 풍부하며, **금융 상품에 대한 반응도**가 높아 효과적인 마케팅 및 상품 전략 적용이 가능함.
#     """)



import os
import streamlit as st
import pandas as pd
import joblib

# ✅ 프로젝트 루트 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📌 모델 로드 함수
@st.cache_resource
def load_model(model_type="rf"):
    return joblib.load(f"model/final_model_{model_type}.pkl")

# 📌 데이터 불러오기 함수 (CSV 활용)
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "data", "clustered_Customer_Churn.csv")  # CSV 파일 경로
    df = pd.read_csv(csv_path)
    return df

# 📌 Streamlit 페이지 설정
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# 📌 메인 페이지 내용
st.title("💡 SKN Bank 고객 이탈 예측 및 금융상품 최적화 시스템")
st.write("""
본 시스템은 **머신러닝을 활용하여 고객 이탈 가능성을 예측**하고,  
맞춤형 금융상품 적용을 통해 **이탈률 감소 및 수익 최적화**를 목표로 합니다.
""")

st.write("### 📌 프로젝트 개요")
st.image("streamlit/img/main_page.jpg", caption="고객 이탈 예측 프로젝트 흐름도", use_container_width=True)

st.markdown("""
**✅ 프로젝트 배경**  
📌 **은행의 고객 이탈 문제 심화**  
- 디지털 금융 서비스 확장으로 인해 전통적인 은행 고객의 **이탈 위험 증가**  
- 신규 고객 유치 비용은 기존 고객 유지 비용의 5배 이상  
- 기존 고객을 효과적으로 유지하는 것이 **수익성 개선의 핵심 요소**  

📌 **데이터 기반 예측의 필요성**  
- 연구(Sage Journal - Reducing Adverse Selection through CRM)에 따르면,  
  대부분의 금융 기관이 **이탈 발생 후 대응하는 사후 관리 방식**을 채택  
- 사전적으로 **고객 이탈을 예측하고, 선제적 대응**이 필요  

📌 **금융상품 적용과 고객 유지 전략 최적화**  
- 고객 이탈을 줄이기 위해 **맞춤형 금융상품 제공**이 중요  
- 이탈 가능성이 높은 고객에게 **최적의 금융상품을 적용하여 고객 유지율 개선**
""")

st.markdown("""
### 📌 기대 효과
| 기대효과 | 내용 |
|---------|------|
| 🔍 고객 이탈 예측 | 머신러닝을 활용해 이탈 가능성이 높은 고객을 조기에 발견, 맞춤형 대응 가능 |
| 📉 이탈률 감소 | 고객 특성에 맞는 금융상품을 적용해 이탈 가능성 감소 |
| 💰 수익 최적화 | 이탈률 감소를 통한 추가 예상 수익 증가 (약 100억 원 이상 기대) |
| 📊 데이터 기반 의사결정 | AI 기반 예측을 통해 고객 관리 및 마케팅 전략 최적화 |

📌 **프로젝트 요약**  
✅ 이탈 예측 모델을 통해 **고위험 고객을 식별**하고, 금융상품 적용 시 이탈률 변화를 분석  
✅ **실제 은행 고객 1500만 명 수준을 고려한 시뮬레이션 수행**  
✅ 클러스터링(군집 분석)을 활용하여 **고객 특성에 따른 맞춤형 금융상품 제공 전략 수립**

🎯 **최종 목표:**  
💡 **데이터 기반 고객 유지 및 금융상품 최적화 전략을 통해 은행의 장기적 성장 지원** 🚀
""")

# 📌 CEO 요구사항 버튼
if st.button("📌 CEO의 요구사항 보기"):
    st.subheader("👨‍💼 CEO의 요구 사항")
    st.markdown("""
    **고객 이탈률을 분석하고 줄일 방법을 찾아라.**  
    - 데이터 기반 솔루션을 활용하여 **고객 유지율을 높이고, 은행의 수익을 최적화**할 것!!
    """)

# 📌 분석 과정 버튼
if st.button("📊 분석 과정 보기"):
    st.header("📊 분석 과정")

    st.subheader("1️⃣ 이탈률을 어떻게 분석할 것인가?")
    st.write("""
    - **고객 데이터를 군집화하여 유형 분류**
    - **군집 특성을 바탕으로 이탈 가능성을 예측**해보는건 어떨까?
    """)

    st.subheader("2️⃣ 클러스터링 방식 선택")
    st.write("""
    - **엘보 기법(Elbow Method)과 실루엣 분석(Silhouette Analysis)을 적용하여 최적 군집 4개**
    """)
    # 📌 클러스터링 최적화 이미지 추가
    cluster_image_path = os.path.join(BASE_DIR, "streamlit/img", "OptimizeCluster.jpg")
    st.image(cluster_image_path, caption="클러스터링 최적화 분석 결과", use_container_width=True)
    
    st.write("""
    - 하지만 4개 군집으로는 **세부 유형 분류가 어려움**
    - 최종적으로 **6개 클러스터로 분류**
    """)
    
    # 📌 클러스터 피쳐 이미지 추가
    ClusterFeature_image_path = os.path.join(BASE_DIR, "streamlit/img", "ClusterFeature.jpg")
    st.image(ClusterFeature_image_path, caption="클러스터 특징", use_container_width=True)

if st.button("🛠 군집 분석"):
    st.subheader("타겟 군집 선택")
    st.write("""
    - 모든 고객의 이탈률을 줄이는 것은 **비용과 시간** 측면에서 비효율적  
    - 금융상품 제공 시 **가장 효과적인 군집**을 선정하는 것이 중요  
      
    📌 **클러스터 분석 결과**
    - **클러스터 1번과 2번**: 은행의 핵심 상품 전략과 연관성이 낮아 이탈 방지를 위한 비용이 클 가능성이 높음  
    - **클러스터 4번(고령층)**: 금융활동이 정체되어 적극적인 마케팅 효과가 제한적  
    - **클러스터 5번(젊은 고객층)**: 장기적인 고객 유치 전략이 필요하지만, 단기 성과가 어려움  
    - **클러스터 0번과 3번**:  
      → **자산이 풍부하며, 금융 상품에 대한 반응도**가 높음  
      → **효과적인 마케팅 및 금융상품 적용 가능**  
      → **VIP 서비스 강화, 맞춤형 금융상품 제공** 등의 전략이 효과적  
    """)

    cluster_image_path = os.path.join(BASE_DIR, "streamlit/img", "ClusterStrategy.jpg")
    st.image(cluster_image_path, caption="클러스터 전략", use_container_width=True)    
    